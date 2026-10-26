package org.beehive.jitllm.model.loader;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.IntFunction;
import org.beehive.jitllm.Options;
import org.beehive.jitllm.auxiliary.RunMetrics;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.tensor.FP16TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.FP32TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.PackedTiles;
import org.beehive.jitllm.backend.tornado.tensor.PackedWeights;
import org.beehive.jitllm.backend.tornado.tensor.Q4_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q4_1TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q4_KTornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q5_KTornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q6_KTornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q8_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jitllm.format.*;
import org.beehive.jitllm.format.DataTypeMapping;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.format.TensorDescriptors;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.provider.ModelProvider;
import org.beehive.jitllm.model.provider.ModelProviders;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.memory.DeviceRetention;
import org.beehive.jitllm.runtime.memory.WeightFootprint;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.ExecutionTarget;
import org.beehive.jitllm.runtime.tensor.TensorDescriptor;
import org.beehive.jitllm.tensor.standard.*;

public final class ModelLoader {

    private ModelLoader() {}

    /**
     * Loads the model the CLI options name.
     *
     * @param options the parsed CLI options containing model path and max token limit
     * @return the loaded {@link Model} instance
     * @throws IOException if the model fails to load
     */
    public static Model loadModel(Options options) throws IOException {
        return loadModel(options.modelPath(), options.maxTokens(), true, options.useTornadovm());
    }

    /** For compatibility with langchain4j and quarkus. */
    public static Model loadModel(
            Path ggufPath, int contextLength, boolean loadWeights, boolean useTornadovm)
            throws IOException {
        if ("false".equalsIgnoreCase(System.getProperty("jitllm.providers"))) {
            throw new UnsupportedOperationException(
                    "-Djitllm.providers=false is no longer supported: every model loads through"
                            + " its provider");
        }
        long start = System.nanoTime();
        ModelSource source = ModelSource.ofFile(ggufPath);
        Model model = loadThroughProvider(source, contextLength, useTornadovm);
        RunMetrics.setLoadDuration(System.nanoTime() - start);
        return model;
    }

    /** A model file's architecture identity and configuration. */
    public record Description(ArchitectureId architecture, Configuration configuration) {}

    /**
     * What a load of this file would be, from its metadata alone: the identity its provider gives
     * it and the configuration it would load with. No tokenizer, tensor or weight is built, which
     * is what lets the memory preflight run before the load it is checking.
     *
     * @param contextLength as for {@link #loadModel(Path, int, boolean, boolean)}
     */
    public static Description describe(Path ggufPath, int contextLength) throws IOException {
        ModelSource source = ModelSource.ofFile(ggufPath);
        ModelProvider provider = ModelProviders.select(source);
        return new Description(
                provider.architecture(source), provider.configuration(source, contextLength));
    }

    /**
     * The discovered provider loads it. Recognition happens once, in the provider that claims the
     * source, and the architecture identity it chooses is the one everything downstream uses.
     */
    private static Model loadThroughProvider(
            ModelSource source, int contextLength, boolean useTornadovm) throws IOException {
        ModelProvider provider = ModelProviders.select(source);
        return provider.load(source, providerBackend(useTornadovm), contextLength);
    }

    /**
     * The identity a provider and its diagnostics see.
     *
     * <p>Was a hardcoded {@code BackendId.CUDA} for every non-CPU load — wrong on Metal and OpenCL
     * alike, and a defect this method's callers ({@code FamilyProviders} and {@code
     * Gemma4Provider}, both only ever comparing the value to {@code BackendId.CPU}) could not
     * observe. When a real accelerator is resolved, this reports it truthfully instead.
     *
     * <p><b>The no-accelerator corner case keeps the old placeholder.</b> If {@code
     * useTornadovm=true} and nothing resolves an accelerator at all — {@code
     * TornadoDevices.current()} itself falls back to the {@code BackendId.CPU} placeholder —
     * reporting that placeholder here would flip every provider's {@code
     * !BackendId.CPU.equals(backend)} check to {@code false} and silently take the CPU path for a
     * caller that asked for the GPU one: a new silent fallback, introduced by fixing an old
     * mislabel. Keeping {@code BackendId.CUDA} here in that one case is exactly the pre-existing
     * behaviour {@code useTornadovm}'s boolean callers already have — unchanged, not newly correct,
     * because correctness for that case is {@code LocalModels}'s explicit-accelerator validation
     * territory, not this compatibility path's.
     *
     * <p>Package-visible for {@code ModelLoaderProviderBackendTest} and its accelerator sibling.
     */
    static BackendId providerBackend(boolean useTornadovm) {
        if (!useTornadovm) {
            return BackendId.CPU;
        }
        BackendId resolved = TornadoDevices.current().id().backend();
        return BackendId.CPU.equals(resolved) ? BackendId.CUDA : resolved;
    }

    /**
     * The weight footprint of a model file, from its descriptors alone.
     *
     * <p>Reads tensor metadata, never tensor data — this is what lets a preflight answer "will it
     * fit" without a multi-gigabyte upload.
     *
     * <p>Lives here because Rule 4 permits the loaders to name GGUF and forbids it to the runtime
     * and the backends. What leaves this method is a neutral {@link WeightFootprint}.
     *
     * <p>The per-layer / global split follows the GGUF convention that a layer's tensors are named
     * {@code blk.N.*}. That is the same convention every loader in this package already relies on
     * to find them, so the two cannot disagree about what a layer owns.
     *
     * <p><b>Sized as the accelerator materializes it, not as the file stores it.</b> This is a
     * device memory plan, and {@link #loadTornadoTensor} materializes a representation the device
     * has no kernel for as Q8_0 — so a Q4_K file's weights occupy roughly twice their file size
     * once loaded. Measuring the file type here under-predicted exactly those models, in the one
     * direction a preflight must never be wrong: {@code MemoryPlanAccuracyAccelTest} exists because
     * a prediction below what is actually allocated admits a load that then dies part-allocated.
     * Found on the real Devstral fixture (Metal parity task 12): a Q4_K 24B predicted 13.5 GiB and
     * died materializing Q8_0 — {@code OutOfMemoryError: Cannot reserve. direct buffer memory at
     * TornadoTensorLoader.dequantizeToQ8_0}. The materialized type comes from {@code
     * DataTypeMapping.materializedType}, the same function {@link #loadTornadoTensor}'s descriptor
     * uses, so the prediction and the allocation cannot disagree about what a tensor becomes. F16,
     * Q8_0 and F32 materialize as themselves, so every tuple measured on CUDA is predicted
     * byte-for-byte as before.
     */
    public static WeightFootprint weightFootprint(Path ggufPath) throws IOException {
        return weightFootprint(ggufPath, DeviceRetention.converting());
    }

    /** {@link #weightFootprint(Path, DeviceRetention)} for a flat set of retained types. */
    public static WeightFootprint weightFootprint(
            Path ggufPath, java.util.Set<DataType> nativeDeviceTypes) throws IOException {
        return weightFootprint(ggufPath, DeviceRetention.retaining(nativeDeviceTypes));
    }

    /**
     * The device footprint of a file's weights, given the representations the target family reads
     * without materializing.
     *
     * <p>Descriptors only — no tensor data is touched, which is what makes this usable before a
     * load rather than after one. The decision is taken per tensor; for a family whose loader
     * decides for the whole model, see {@link #weightFootprint(Path, Function)}.
     *
     * @param retention what representation each tensor will occupy on the device
     */
    public static WeightFootprint weightFootprint(Path ggufPath, DeviceRetention retention)
            throws IOException {
        return weightFootprint(GGUF.loadGGUFMetadata(ggufPath), retention);
    }

    /**
     * {@link #weightFootprint(Path, DeviceRetention)} with the retention decided from the file
     * itself: {@code retention} is given every tensor's representation, by name, and answers what
     * each will occupy. That is what a loader that decides for the whole model reads — Llama keeps
     * Q4_0 only when every per-layer projection is Q4_0, and materializes all of them otherwise —
     * so the prediction follows the same rule as the load.
     *
     * @param retention the device representation of each tensor, given every tensor's
     *     representation in the file
     */
    public static WeightFootprint weightFootprint(
            Path ggufPath, Function<Map<String, DataType>, DeviceRetention> retention)
            throws IOException {
        GGUF gguf = GGUF.loadGGUFMetadata(ggufPath);
        Map<String, DataType> sourceTypes = new HashMap<>();
        for (GGUF.GGUFTensorInfo info : gguf.getTensorInfos().values()) {
            sourceTypes.put(info.name(), DataTypeMapping.sourceType(info.ggmlType()));
        }
        return weightFootprint(gguf, retention.apply(sourceTypes));
    }

    private static WeightFootprint weightFootprint(GGUF gguf, DeviceRetention retention) {
        long perLayer = 0;
        long global = 0;
        int perLayerTensors = 0;
        int globalTensors = 0;
        for (GGUF.GGUFTensorInfo info : gguf.getTensorInfos().values()) {
            if (info.name().equals("rope_freqs.weight")) {
                continue; // not materialized — every loader skips it
            }
            long elements = 1L;
            for (int d : info.dimensions()) {
                elements *= d;
            }
            DataType source = DataTypeMapping.sourceType(info.ggmlType());
            // Per tensor, by name and representation: a model is not one dtype, and support can
            // differ by role as well as by format.
            DataType materialized = retention.deviceType(info.name(), source);
            long bytes = TensorDescriptors.layoutOf(materialized).byteSize(elements);
            if (info.name().startsWith("blk.")) {
                perLayer += bytes;
                perLayerTensors++;
            } else {
                global += bytes;
                globalTensors++;
            }
        }
        return new WeightFootprint(perLayer, perLayerTensors, global, globalTensors);
    }

    /**
     * The bytes of the largest per-layer projection a load would pack, given the representations
     * the family packs: a two-dimensional {@code blk.*} tensor of one of {@code packed} whose shape
     * packs ({@link PackedTiles#fits}). Zero when there is none.
     *
     * <p>Descriptors only, like {@link #weightFootprint}. It is the size of the scratch buffer the
     * packed weights are repacked through on the device, which is as large as the largest of them.
     */
    public static long packedProjectionBytes(Path ggufPath, java.util.Set<DataType> packed)
            throws IOException {
        if (packed.isEmpty()) {
            return 0;
        }
        GGUF gguf = GGUF.loadGGUFMetadata(ggufPath);
        long largest = 0;
        for (GGUF.GGUFTensorInfo info : gguf.getTensorInfos().values()) {
            DataType source = DataTypeMapping.sourceType(info.ggmlType());
            int[] shape = info.dimensions();
            if (!info.name().startsWith("blk.")
                    || !packed.contains(source)
                    || shape.length != 2
                    || !PackedTiles.fits(shape[1], shape[0])) {
                continue;
            }
            long bytes = TensorDescriptors.layoutOf(source).byteSize((long) shape[0] * shape[1]);
            largest = Math.max(largest, bytes);
        }
        return largest;
    }

    public static FloatTensor loadTensor(GGMLTensorEntry entry) {
        TensorDescriptor descriptor = TensorDescriptors.describeSource(entry);
        int size = descriptor.shape().elementCountAsInt(descriptor.name());
        MemorySegment data = entry.memorySegment();
        return switch (entry.ggmlType()) {
            case F32 -> new FP32FloatTensor(size, data);
            case Q8_0 -> new Q8_0FloatTensor(size, data);
            case Q4_0 -> new Q4_0FloatTensor(size, data);
            case Q4_1 -> new Q4_1FloatTensor(size, data);
            case Q4_K -> new Q4_KFloatTensor(size, data);
            case Q5_K -> new Q5_KFloatTensor(size, data);
            case Q6_K -> new Q6_KFloatTensor(size, data);
            case F16 -> new FP16FloatTensor(size, data);
            case BF16 -> new BF16FloatTensor(size, data);
            default ->
                    throw new UnsupportedOperationException(
                            "Quantization format " + entry.ggmlType());
        };
    }

    /** Dispatcher method for loading a standard tensor array based on type. Used in CPU-path. */
    public static FloatTensor[] loadArrayOfTensors(
            int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTensor(getTensorEntry.apply(i));
        }
        return array;
    }

    /** Loads a tensor for the device <b>retaining Q4_K</b> rather than materializing it as Q8_0. */
    public static TornadoTensor loadTornadoTensorRetainingQ4_K(GGMLTensorEntry entry) {
        if (entry.ggmlType() == GGMLType.Q4_K) {
            return Q4_KTornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
        }
        if (entry.ggmlType() == GGMLType.Q6_K) {
            return Q6_KTornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
        }
        return loadTornadoTensor(entry);
    }

    /**
     * Loads a tensor for the device <b>retaining Q4_0</b> rather than materializing it as Q8_0.
     *
     * <p>Separate from {@link #loadTornadoTensorRetainingQ4_K} rather than one helper that retains
     * everything with a kernel: which representations a family can read is a property of that
     * family's layer graph, and a tensor retained in a format the graph has no kernel for would be
     * read as the format it is not — fluent output, wrong numbers. A loader opts into exactly what
     * its own layers dispatch on.
     */
    public static TornadoTensor loadTornadoTensorRetainingQ4_0(GGMLTensorEntry entry) {
        if (entry.ggmlType() == GGMLType.Q4_0) {
            return Q4_0TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
        }
        return loadTornadoTensor(entry);
    }

    /**
     * Loads a tensor for the device in <b>whatever representation the file gave it</b>, for every
     * quantization the backend has device storage and kernels for.
     *
     * <p>The general form of the two helpers above, and the one a family uses when its layer graph
     * dispatches per tensor rather than assuming one representation. Nothing is converted: a Q4_1
     * tensor stays Q4_1, a Q5_K tensor stays Q5_K, and a representation with no device storage is
     * an error here rather than a quiet promotion to Q8_0.
     *
     * @throws ModelLoadException if the file holds a representation the device cannot store
     */
    public static TornadoTensor loadTornadoTensorNative(GGMLTensorEntry entry) {
        return switch (entry.ggmlType()) {
            case F32 -> FP32TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case F16 -> FP16TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case Q8_0 -> Q8_0TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case Q4_0 -> Q4_0TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case Q4_1 -> Q4_1TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case Q4_K -> Q4_KTornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case Q5_K -> Q5_KTornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case Q6_K -> Q6_KTornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case BF16 -> TornadoTensorLoader.convertBF16ToFP16(rawTensorData(entry));
            default ->
                    throw new ModelLoadException(
                            DiagnosticCode.MODEL_MALFORMED.prefix()
                                    + entry.name()
                                    + " is "
                                    + entry.ggmlType()
                                    + ", for which this backend has no device storage. It is not"
                                    + " converted to Q8_0: a representation the device cannot hold"
                                    + " is a gap to fill, not something to promote silently.");
        };
    }

    /** {@link #loadArrayOfTornadoTensors} keeping every representation as the file gave it. */
    public static TornadoTensor[] loadArrayOfTornadoTensorsNative(
            int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        TornadoTensor[] array = new TornadoTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTornadoTensorNative(getTensorEntry.apply(i));
        }
        return array;
    }

    /** {@link #loadArrayOfTornadoTensors} that retains Q4_0. */
    public static TornadoTensor[] loadArrayOfTornadoTensorsRetainingQ4_0(
            int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        TornadoTensor[] array = new TornadoTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTornadoTensorRetainingQ4_0(getTensorEntry.apply(i));
        }
        return array;
    }

    public static TornadoTensor loadTornadoTensor(GGMLTensorEntry entry) {
        // Describe first: the descriptor states what this tensor becomes on the device — including
        // the Q8_0 materialization for representations with no kernel — and validates the element
        // count before any storage is touched. It holds no data, so nothing is copied for it.
        TensorDescriptor descriptor = TensorDescriptors.describe(entry, ExecutionTarget.GPU);
        descriptor.shape().elementCountAsInt(descriptor.name());
        return switch (entry.ggmlType()) {
            case F32 -> FP32TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case F16 -> FP16TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case BF16 -> TornadoTensorLoader.convertBF16ToFP16(rawTensorData(entry));
            case Q8_0 -> Q8_0TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
                // A representation the device has no kernel for is materialized as Q8_0 at load
                // The conversion reads through the CPU tensor for that format, which
                // already knows how to decode it.
            case Q4_0, Q4_1, Q4_K, Q5_K, Q6_K ->
                    TornadoTensorLoader.dequantizeToQ8_0(rawTensorData(entry));
            default ->
                    throw new UnsupportedOperationException(
                            "Quantization format " + entry.ggmlType());
        };
    }

    /**
     * The entry's tensor data, read as a CPU tensor, past the device array header.
     *
     * <p>An entry loaded for the device is prefixed with TornadoVM's array header; how wide that is
     * is the backend's knowledge, so the slice comes from there.
     */
    private static FloatTensor rawTensorData(GGMLTensorEntry entry) {
        GGMLTensorEntry dataEntry =
                new GGMLTensorEntry(
                        entry.mappedFile(),
                        entry.name(),
                        entry.ggmlType(),
                        entry.shape(),
                        TornadoTensorLoader.withoutArrayHeader(entry.memorySegment()));
        return loadTensor(dataEntry);
    }

    /**
     * {@link #loadArrayOfTornadoTensors} that retains Q4_K — see {@link
     * #loadTornadoTensorRetainingQ4_K}. For the per-layer weights of a family that has Q4_K
     * kernels.
     */
    public static TornadoTensor[] loadArrayOfTornadoTensorsRetainingQ4_K(
            int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        TornadoTensor[] array = new TornadoTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTornadoTensorRetainingQ4_K(getTensorEntry.apply(i));
        }
        return array;
    }

    public static TornadoTensor[] loadArrayOfTornadoTensors(
            int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        TornadoTensor[] array = new TornadoTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTornadoTensor(getTensorEntry.apply(i));
        }
        return array;
    }

    /**
     * Q4_0 projection packing, when {@code pack} ({@link PackedWeights#packQ4}), for the loaders
     * whose layer graphs read packed Q4_0 tiles: each retained Q4_0 projection whose shape packs is
     * recorded as packed, and repacked on the GPU once uploaded (see {@link PackedTiles}). A
     * projection of another type — the Q4_1 {@code ffn_down} of some layers, a materialized trunk —
     * is left as it is.
     */
    public static TornadoTensor[] packQ4(
            TornadoTensor[] tensors, IntFunction<GGMLTensorEntry> entries, boolean pack) {
        if (!pack) {
            return tensors;
        }
        for (int i = 0; i < tensors.length; i++) {
            GGMLTensorEntry entry = entries.apply(i);
            TornadoTensor tensor = tensors[i];
            if (entry == null
                    || tensor == null
                    || entry.ggmlType() != GGMLType.Q4_0
                    || tensor.dataType() != DataType.Q4_0
                    || entry.shape().length != 2) {
                continue;
            }
            int cols = entry.shape()[0];
            int rows = entry.shape()[1];
            if (PackedTiles.fits(rows, cols)) {
                tensors[i] =
                        new Q4_0TornadoTensor(tensor.asByteArray(), new PackedTiles(rows, cols));
            }
        }
        return tensors;
    }
}
