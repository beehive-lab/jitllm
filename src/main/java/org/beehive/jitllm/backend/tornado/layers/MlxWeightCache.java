package org.beehive.jitllm.backend.tornado.layers;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.zip.CRC32;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

// @formatter:off
/**
 * MLX affine repacks kept on disk and mapped, so a large model's MLX weights cost no anonymous host
 * memory and are repacked once.
 *
 * <p><b>Why.</b> The repack of a 24B Q4_K model is about 15 GB. Held in ordinary native arrays it
 * stays resident on the host beside its device copy and the model's own weights, which a 48 GB Mac
 * cannot hold. Written to a file and mapped {@link FileChannel.MapMode#PRIVATE} — as the GGUF
 * loader maps tensors — its pages are clean file pages the system can drop once the device has its
 * copy.
 *
 * <p><b>Keyed by content, not by path.</b> A file's name is a fingerprint of its source tensors:
 * their types, sizes and a CRC32 over samples spread through each. A model file replaced under the
 * same name gets a new key; the key is also stored in the file and checked when it is mapped, so a
 * truncated or foreign file is rebuilt rather than read. Written to a temporary file and renamed,
 * so a crash leaves no half-written entry.
 *
 * <p>The directory is {@code -Djitllm.mlx.cacheDir}, by default {@code ~/.cache/jitllm/mlx}.
 */
// @formatter:on
public final class MlxWeightCache {

    private static final int MAGIC = 0x584d4c4a; // "JLMX"
    private static final int VERSION = 1;
    private static final long ALIGN = 16384;
    private static final int HEADER_BYTES = 64;
    private static final int SAMPLES = 64;
    private static final int SAMPLE_BYTES = 4096;

    /**
     * Every matrix this process has mapped, by key: one Java object per matrix, so the graphs that
     * read it — the prefill graph that uploads it and the decode graph that binds it from there —
     * name the same device buffer. A second mapping would be a second device copy.
     */
    private static final java.util.concurrent.ConcurrentHashMap<Long, MlxPrefillProjections.Matrix>
            MAPPED = new java.util.concurrent.ConcurrentHashMap<>();

    private final Path directory;

    public MlxWeightCache(Path directory) {
        this.directory = directory;
    }

    /** The cache in {@code -Djitllm.mlx.cacheDir}, or {@code ~/.cache/jitllm/mlx}. */
    public static MlxWeightCache standard() {
        String dir = System.getProperty("jitllm.mlx.cacheDir");
        return new MlxWeightCache(
                dir != null
                        ? Path.of(dir)
                        : Path.of(System.getProperty("user.home"), ".cache", "jitllm", "mlx"));
    }

    /**
     * {@code tensors}, rows stacked in order, in MLX's affine format ({@link
     * MlxPrefillProjections#affine}), from the cache if it holds them and repacked into it if not.
     */
    MlxPrefillProjections.Matrix affine(TornadoTensor... tensors) {
        long key = key(tensors);
        MlxPrefillProjections.Matrix known = MAPPED.get(key);
        if (known != null) {
            return known;
        }
        MlxPrefillProjections.Matrix matrix = load(key, tensors);
        known = MAPPED.putIfAbsent(key, matrix);
        return known != null ? known : matrix;
    }

    private MlxPrefillProjections.Matrix load(long key, TornadoTensor... tensors) {
        Path file = directory.resolve(String.format("%016x.mlx", key));
        MlxPrefillProjections.Matrix cached = map(file, key);
        if (cached != null) {
            return cached;
        }
        write(file, key, MlxPrefillProjections.affine(tensors));
        cached = map(file, key);
        if (cached == null) {
            throw new IllegalStateException("the MLX weight cache could not read back " + file);
        }
        return cached;
    }

    /** A fingerprint of the tensors: types, sizes, and a CRC32 over samples spread through each. */
    static long key(TornadoTensor... tensors) {
        CRC32 crc = new CRC32();
        ByteBuffer meta = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
        long hash = VERSION;
        for (TornadoTensor t : tensors) {
            MemorySegment src = t.asByteArray().getSegment();
            long size = src.byteSize();
            meta.clear();
            meta.putInt(t.dataType().ordinal()).putLong(size).putInt(tensors.length);
            crc.update(meta.array());
            byte[] sample = new byte[SAMPLE_BYTES];
            for (int i = 0; i < SAMPLES; i++) {
                long offset = size <= SAMPLE_BYTES ? 0 : (size - SAMPLE_BYTES) / (SAMPLES - 1) * i;
                int length = (int) Math.min(SAMPLE_BYTES, size - offset);
                MemorySegment.copy(src, ValueLayout.JAVA_BYTE, offset, sample, 0, length);
                crc.update(sample, 0, length);
            }
            hash = hash * 0x9E3779B97F4A7C15L + crc.getValue() + size;
        }
        return hash;
    }

    private void write(Path file, long key, MlxPrefillProjections.Matrix m) {
        try {
            Files.createDirectories(directory);
            Path tmp = Files.createTempFile(directory, "partial-", ".mlx");
            try (FileChannel out = FileChannel.open(tmp, StandardOpenOption.WRITE)) {
                MemorySegment[] parts = {
                    m.q().getSegment(), m.scales().getSegment(), m.biases().getSegment()
                };
                ByteBuffer header =
                        ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
                header.putInt(MAGIC).putInt(VERSION).putLong(key).putInt(m.bits());
                for (MemorySegment p : parts) {
                    header.putLong(p.byteSize());
                }
                header.rewind();
                writeFully(out, header, 0);
                long offset = ALIGN;
                for (MemorySegment p : parts) {
                    // Each array is laid out as TornadoVM's header, then its data.
                    writeFully(out, p.asByteBuffer(), offset + TornadoNativeArray.ARRAY_HEADER);
                    offset = align(offset + TornadoNativeArray.ARRAY_HEADER + p.byteSize());
                }
            }
            Files.move(
                    tmp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new UncheckedIOException("the MLX weight cache could not write " + file, e);
        }
    }

    /** The matrix in {@code file}, or null if there is none or it is not the one keyed. */
    private MlxPrefillProjections.Matrix map(Path file, long key) {
        if (!Files.isRegularFile(file)) {
            return null;
        }
        // Read-write only because a private mapping requires it; the file is never written.
        try (FileChannel in =
                FileChannel.open(file, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            in.read(header, 0);
            header.flip();
            if (header.remaining() < HEADER_BYTES
                    || header.getInt() != MAGIC
                    || header.getInt() != VERSION
                    || header.getLong() != key) {
                return null;
            }
            int bits = header.getInt();
            long[] sizes = {header.getLong(), header.getLong(), header.getLong()};
            long offset = ALIGN;
            long[] offsets = new long[3];
            for (int i = 0; i < 3; i++) {
                offsets[i] = offset;
                offset = align(offset + TornadoNativeArray.ARRAY_HEADER + sizes[i]);
            }
            if (in.size() < offsets[2] + TornadoNativeArray.ARRAY_HEADER + sizes[2]) {
                return null;
            }
            // Private: TornadoVM writes the element count into each array's header, which copies
            // that one page and leaves the file as it is.
            MemorySegment mapped =
                    in.map(FileChannel.MapMode.PRIVATE, 0, in.size(), Arena.ofAuto());
            IntArray q =
                    IntArray.fromSegmentShallow(
                            mapped.asSlice(offsets[0], TornadoNativeArray.ARRAY_HEADER + sizes[0]));
            HalfFloatArray scales =
                    HalfFloatArray.fromSegmentShallow(
                            mapped.asSlice(offsets[1], TornadoNativeArray.ARRAY_HEADER + sizes[1]));
            HalfFloatArray biases =
                    HalfFloatArray.fromSegmentShallow(
                            mapped.asSlice(offsets[2], TornadoNativeArray.ARRAY_HEADER + sizes[2]));
            return new MlxPrefillProjections.Matrix(null, q, scales, biases, bits);
        } catch (IOException e) {
            return null;
        }
    }

    private static void writeFully(FileChannel out, ByteBuffer buffer, long position)
            throws IOException {
        while (buffer.hasRemaining()) {
            position += out.write(buffer, position);
        }
    }

    private static long align(long offset) {
        return (offset + ALIGN - 1) / ALIGN * ALIGN;
    }
}
