package org.beehive.jllm.format;

import static java.nio.file.StandardOpenOption.READ;
import static java.nio.file.StandardOpenOption.WRITE;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jllm.auxiliary.Pair;
import org.beehive.jllm.runtime.backend.Device;

public final class GGUF {
    private static FileChannel fileChannel;
    private static final int GGUF_MAGIC = 0x46554747;
    private static final int DEFAULT_ALIGNMENT = 32; // must be a power of 2
    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(2, 3);
    private final ByteBuffer BB_1 = ByteBuffer.allocate(Byte.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_2 = ByteBuffer.allocate(Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_4 =
            ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private int magic;
    private int version;
    private int tensorCount; // uint64_t
    private int alignment;
    private int metadata_kv_count; // uint64_t
    private Map<String, Object> metadata;
    private Map<String, GGUFTensorInfo> tensorInfos;
    private long tensorDataOffset;

    public static GGUF loadGGUFMetadata(Path modelPath) throws IOException {

        // file existence check
        if (!Files.exists(modelPath)) {
            throw new FileNotFoundException("Model file not found: " + modelPath);
        }

        // Open file
        try {
            fileChannel = FileChannel.open(modelPath, READ, WRITE);
            // Ensure we start reading from the beginning of the file
            fileChannel.position(0);
        } catch (Exception e) {
            throw new RuntimeException("Failed to open file channel for " + modelPath, e);
        }

        // Read and store the gguf metadata
        try {
            GGUF gguf = new GGUF();
            // The header of the file.
            gguf.readHeader(fileChannel); // gguf_header_t header;
            // Tensor infos, which can be used to locate the tensor data.
            // gguf_tensor_info_t tensor_infos[header.tensor_count];
            gguf.tensorInfos = HashMap.newHashMap(gguf.tensorCount);
            for (int i = 0; i < gguf.tensorCount; ++i) {
                GGUF.GGUFTensorInfo ti = gguf.readTensorInfo(fileChannel);
                assert !gguf.tensorInfos.containsKey(ti.name);
                gguf.tensorInfos.put(ti.name, ti);
            }
            // Padding to the nearest multiple of `ALIGNMENT`.
            // uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)];
            long _padding =
                    (gguf.getAlignment() - (fileChannel.position() % gguf.getAlignment()))
                            % gguf.getAlignment();
            fileChannel.position(fileChannel.position() + _padding);
            // Tensor data.
            //
            // This is arbitrary binary data corresponding to the weights of the model. This data
            // should be close
            // or identical to the data in the original model file, but may be different due to
            // quantization or
            // other optimizations for inference. Any such deviations should be recorded in the
            // metadata or as
            // part of the architecture definition.
            //
            // Each tensor's data must be stored within this array, and located through its
            // `tensor_infos` entry.
            // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space
            // between tensors
            // should be padded to `ALIGNMENT` bytes.
            // uint8_t tensor_data[];
            gguf.tensorDataOffset = fileChannel.position();
            return gguf;
        } catch (Exception e) {
            throw new RuntimeException(
                    "Unexpected error while loading GGUF model from " + modelPath, e);
        }
    }

    /**
     * Loads tensor data from a given file channel based on the tensor metadata information. The
     * mapping is read-only and creates standard memory segments for each tensor.
     *
     * @param fileChannel the channel from which tensor storage is read
     * @param tensorDataOffset the absolute byte offset of the GGUF tensor-data section
     * @param tensorInfos metadata describing all GGUF tensors
     * @return a map from tensor name to {@link GGMLTensorEntry} containing standard memory segments
     *     for each tensor
     * @throws IOException if memory mapping fails or the channel cannot be read
     */
    public static Map<String, GGMLTensorEntry> loadTensorsStandard(
            FileChannel fileChannel, long tensorDataOffset, Map<String, GGUFTensorInfo> tensorInfos)
            throws IOException {
        Arena arena = Arena.ofAuto();

        // absolute file offset where the tensor-data section begins
        long mappingOffset = tensorDataOffset;
        // size of the entire tensor-data section
        long mappingSize = fileChannel.size() - tensorDataOffset;

        MemorySegment tensorData =
                fileChannel.map(FileChannel.MapMode.READ_ONLY, mappingOffset, mappingSize, arena);

        Map<String, GGMLTensorEntry> tensorEntries = HashMap.newHashMap(tensorInfos.size());

        for (Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet()) {
            GGUFTensorInfo ti = entry.getValue();

            // skip rope_freqs.weight (not needed for inference)
            if (ti.name().equals("rope_freqs.weight")) {
                continue;
            }

            // Long arithmetic: some tensors (e.g. Gemma4's per-layer token embedding table) exceed
            // Integer.MAX_VALUE elements/bytes, which would overflow the int-based
            // FloatTensor.numberOfElements/GGMLType.byteSizeFor. Such tensors are read directly via
            // long-offset MemorySegment access rather than wrapped in a FloatTensor, so only the
            // slice size (which MemorySegment#asSlice accepts as a long) matters here.
            long sizeInBytes = tensorByteSize(ti);

            // per-tensor slice offset; ti.offset() is relative to tensor-data start
            long offset = ti.offset();

            // per-tensor slice segment
            MemorySegment memorySegment = tensorData.asSlice(offset, sizeInBytes);

            tensorEntries.put(
                    ti.name(),
                    new GGMLTensorEntry(
                            tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), memorySegment));
        }
        return tensorEntries;
    }

    /**
     * Loads GGUF tensor data in a layout a device backend's native arrays can be built over.
     *
     * <p>This method parses the GGUF tensor list and memory-maps each tensor directly from the
     * underlying {@link FileChannel}. A backend's native array carries a header in front of its
     * elements, so each tensor is mapped using {@link FileChannel.MapMode#PRIVATE} starting {@code
     * device.nativeArrayHeaderBytes()} before the actual tensor position, providing a writable
     * header region without modifying the underlying GGUF file.
     *
     * <p><b>How big that header is is not GGUF's fact</b>. This method used to read {@code
     * TornadoNativeArray.ARRAY_HEADER}, which is how a file-format parser came to name a backend
     * runtime type — the last entry on Rule 1's allowlist. It now asks the resolved {@link Device},
     * the same way every other backend fact is obtained, and parsing itself is unchanged: the
     * header is zeroed exactly as before and the mapped bytes are the same bytes.
     *
     * @param fileChannel the channel from which tensor storage is read
     * @param tensorDataOffset the absolute byte offset of the GGUF tensor-data section
     * @param tensorInfos metadata describing all GGUF tensors
     * @param device the resolved device whose native-array layout the mapping must suit
     * @return a map from tensor name to {@link GGMLTensorEntry} containing memory segments laid out
     *     for that device's native arrays
     * @throws IOException if memory mapping fails or the channel cannot be read
     */
    public static Map<String, GGMLTensorEntry> loadTensorsTornado(
            FileChannel fileChannel,
            long tensorDataOffset,
            Map<String, GGUFTensorInfo> tensorInfos,
            Device device)
            throws IOException {

        Arena arena = Arena.ofAuto();
        Map<String, GGMLTensorEntry> tensorEntries = HashMap.newHashMap(tensorInfos.size());

        for (Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet()) {
            GGUFTensorInfo ti = entry.getValue();

            // skip rope_freqs.weight (not required for inference)
            if (ti.name().equals("rope_freqs.weight")) {
                continue;
            }

            // see loadTensorsStandard for why this uses long arithmetic instead of
            // FloatTensor.numberOfElements/GGMLType.byteSizeFor
            long sizeInBytes = tensorByteSize(ti);

            // absolute tensor offset - relative to start of the file
            long mappingOffset = tensorDataOffset + ti.offset();

            // create memory segment in the backend's native-array layout:
            // header (16 bytes on TornadoVM) + tensor data
            long headerBytes = device.nativeArrayHeaderBytes();

            // start the header's width before the tensor position to include header space
            long offset = mappingOffset - headerBytes;
            long size = sizeInBytes + headerBytes;
            MemorySegment memorySegment =
                    fileChannel.map(FileChannel.MapMode.PRIVATE, offset, size, arena);

            // zero out the 16-byte header
            for (int i = 0; i < headerBytes; i++) {
                memorySegment.set(ValueLayout.JAVA_BYTE, i, (byte) 0);
            }

            // store the device-compatible segment
            tensorEntries.put(
                    ti.name(),
                    new GGMLTensorEntry(
                            memorySegment,
                            ti.name(),
                            ti.ggmlType(),
                            ti.dimensions(),
                            memorySegment));
        }
        return tensorEntries;
    }

    /**
     * Computes a tensor's byte size with {@code long} arithmetic, avoiding int-overflow for very
     * large tensors.
     */
    private static long tensorByteSize(GGUFTensorInfo ti) {
        long numberOfElements = 1L;
        for (int dimension : ti.dimensions()) {
            numberOfElements *= dimension;
        }
        long typeSize = ti.ggmlType().getTypeSize();
        long blockSize = ti.ggmlType().getBlockSize();
        assert (numberOfElements * typeSize) % blockSize == 0;
        return numberOfElements * typeSize / blockSize;
    }

    public Map<String, GGUFTensorInfo> getTensorInfos() {
        return tensorInfos;
    }

    public long getTensorDataOffset() {
        return tensorDataOffset;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public FileChannel getFileChannel() {
        return fileChannel;
    }

    private GGMLType readGGMLType(FileChannel fileChannel) throws IOException {
        int ggmlTypeId = readInt(fileChannel); // ggml_type type;
        return GGMLType.fromId(ggmlTypeId);
    }

    private GGUF.GGUFTensorInfo readTensorInfo(FileChannel fileChannel) throws IOException {
        // The name of the tensor. It is a standard GGUF string, with the caveat that
        // it must be at most 64 bytes long.
        String name = readString(fileChannel); // gguf_string_t name;
        assert name.length() <= 64;
        // The number of dimensions in the tensor.
        // Currently at most 4, but this may change in the future.
        int n_dimensions = readInt(fileChannel); // uint32_t n_dimensions;
        assert n_dimensions <= 4;
        // The dimensions of the tensor.
        int[] dimensions = new int[n_dimensions]; // uint64_t dimensions[n_dimensions];
        for (int i = 0; i < n_dimensions; ++i) {
            dimensions[i] = Math.toIntExact(readLong(fileChannel));
        }
        // The type of the tensor.
        GGMLType ggmlType = readGGMLType(fileChannel); // ggml_type type;
        // The offset of the tensor's data in this file in bytes.
        // This offset is relative to `tensor_data`, not to the start
        // of the file, to make it easier for writers to write the file.
        // Readers should consider exposing this offset relative to the
        // file to make it easier to read the data.
        // Must be a multiple of `ALIGNMENT`.
        long offset = readLong(fileChannel); // uint64_t offset;
        assert offset % getAlignment() == 0;
        return new GGUF.GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }

    private String readString(FileChannel fileChannel) throws IOException {
        // A string in GGUF.
        // The length of the string, in bytes.
        int len = Math.toIntExact(readLong(fileChannel)); // uint64_t len;
        // The string as a UTF-8 non-null-terminated string.
        byte[] bytes = new byte[len]; // char string[len];
        int bytesRead = fileChannel.read(ByteBuffer.wrap(bytes));
        assert len == bytesRead;
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private Pair<String, Object> readKeyValuePair(FileChannel fileChannel) throws IOException {
        // The key of the metadata. It is a standard GGUF string, with the following caveats:
        // - It must be a valid ASCII string.
        // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated
        // by a `.`.
        // - It must be at most 2^16-1/65535 bytes long.
        // Any keys that do not follow these rules are invalid.
        String key = readString(fileChannel); // gguf_string_t key;
        assert key.length() < (1 << 16);
        assert key.codePoints()
                .allMatch(
                        cp ->
                                ('a' <= cp && cp <= 'z')
                                        || ('0' <= cp && cp <= '9')
                                        || cp == '_'
                                        || cp == '.');
        Object value = readMetadataValue(fileChannel);
        return new Pair<>(key, value);
    }

    private Object readMetadataValue(FileChannel fileChannel) throws IOException {
        // The type of the value.
        // Must be one of the `gguf_metadata_value_type` values.
        MetadataValueType value_type =
                readMetadataValueType(fileChannel); // gguf_metadata_value_type value_type;
        // The value.
        return readMetadataValueOfType(value_type, fileChannel); // gguf_metadata_value_t value;
    }

    void readHeader(FileChannel fileChannel) throws IOException {
        // Magic number to announce that this is a GGUF file.
        // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
        // Your executor might do little-endian byte order, so it might be
        // check for 0x46554747 and letting the endianness cancel out.
        // Consider being *very* explicit about the byte order here.
        this.magic = readInt(fileChannel); //    uint32_t magic;
        if (magic != GGUF_MAGIC) {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }
        // The version of the format implemented.
        // Must be `3` for version described in this spec.
        //
        // This version should only be increased for structural changes to the format.
        // Changes that do not affect the structure of the file should instead update the metadata
        // to signify the change.
        this.version = readInt(fileChannel); // uint32_t version;
        if (!SUPPORTED_GGUF_VERSIONS.contains(version)) {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }
        // The number of tensors in the file. This is explicit, instead of being included in the
        // metadata, to ensure it is always present for loading the tensors.
        this.tensorCount = Math.toIntExact(readLong(fileChannel)); // uint64_t tensor_count;
        // The number of metadata key-value pairs.
        this.metadata_kv_count =
                Math.toIntExact(readLong(fileChannel)); // uint64_t metadata_kv_count;
        // The metadata key-value pairs.
        // gguf_metadata_kv_t metadata_kv[metadata_kv_count];

        this.metadata = HashMap.newHashMap(metadata_kv_count);
        for (int i = 0; i < metadata_kv_count; ++i) {
            Pair<String, Object> keyValue = readKeyValuePair(fileChannel);
            assert !metadata.containsKey(keyValue.first());
            metadata.put(keyValue.first(), keyValue.second());
        }
    }

    private Object readArray(FileChannel fileChannel) throws IOException {
        // Any value type is valid, including arrays.
        MetadataValueType value_type =
                readMetadataValueType(fileChannel); // gguf_metadata_value_type type;
        // Number of elements, not bytes
        int len = Math.toIntExact(readLong(fileChannel)); // uint64_t len;
        // The array of values.
        // gguf_metadata_value_t array[len];
        switch (value_type) {
            case UINT8, INT8 -> {
                byte[] bytes = new byte[len];
                for (int i = 0; i < len; ++i) {
                    bytes[i] = readByte(fileChannel);
                }
                return bytes;
            }
            case UINT16, INT16 -> {
                short[] shorts = new short[len];
                for (int i = 0; i < len; ++i) {
                    shorts[i] = readShort(fileChannel);
                }
                return shorts;
            }
            case UINT32, INT32 -> {
                int[] ints = new int[len];
                for (int i = 0; i < len; ++i) {
                    ints[i] = readInt(fileChannel);
                }
                return ints;
            }
            case FLOAT32 -> {
                float[] floats = new float[len];
                for (int i = 0; i < len; ++i) {
                    floats[i] = readFloat(fileChannel);
                }
                return floats;
            }
            case BOOL -> {
                boolean[] booleans = new boolean[len];
                for (int i = 0; i < len; ++i) {
                    booleans[i] = readBoolean(fileChannel);
                }
                return booleans;
            }
            case STRING -> {
                String[] strings = new String[len];
                for (int i = 0; i < len; ++i) {
                    strings[i] = readString(fileChannel);
                }
                return strings;
            }
            case ARRAY -> {
                Object[] arrays = new Object[len];
                for (int i = 0; i < len; ++i) {
                    arrays[i] = readArray(fileChannel);
                }
                return arrays;
            }
            default -> throw new UnsupportedOperationException("read array of " + value_type);
        }
    }

    private Object readMetadataValueOfType(MetadataValueType valueType, FileChannel fileChannel)
            throws IOException {
        return switch (valueType) {
            case UINT8, INT8 -> readByte(fileChannel);
            case UINT16, INT16 -> readShort(fileChannel);
            case UINT32, INT32 -> readInt(fileChannel);
            case FLOAT32 -> readFloat(fileChannel);
            case UINT64, INT64 -> readLong(fileChannel);
            case FLOAT64 -> readDouble(fileChannel);
            case BOOL -> readBoolean(fileChannel);
            case STRING -> readString(fileChannel);
            case ARRAY -> readArray(fileChannel);
        };
    }

    private byte readByte(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_1);
        assert bytesRead == 1;
        return BB_1.clear().get(0);
    }

    private boolean readBoolean(FileChannel fileChannel) throws IOException {
        return readByte(fileChannel) != 0;
    }

    private short readShort(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_2);
        assert bytesRead == 2;
        return BB_2.clear().getShort(0);
    }

    private int readInt(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_4);
        assert bytesRead == 4;
        return BB_4.clear().getInt(0);
    }

    private long readLong(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_8);
        assert bytesRead == 8;
        return BB_8.clear().getLong(0);
    }

    private float readFloat(FileChannel fileChannel) throws IOException {
        return Float.intBitsToFloat(readInt(fileChannel));
    }

    private double readDouble(FileChannel fileChannel) throws IOException {
        return Double.longBitsToDouble(readLong(fileChannel));
    }

    private MetadataValueType readMetadataValueType(FileChannel fileChannel) throws IOException {
        int index = readInt(fileChannel);
        return MetadataValueType.fromIndex(index);
    }

    public int getAlignment() {
        if (alignment != 0) {
            return alignment;
        }
        alignment = (int) metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        return alignment;
    }

    public record GGUFTensorInfo(String name, int[] dimensions, GGMLType ggmlType, long offset) {}
}
