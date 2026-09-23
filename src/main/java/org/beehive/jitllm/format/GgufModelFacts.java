package org.beehive.jllm.format;

import java.io.IOException;
import java.nio.file.Path;

/**
 * What a GGUF file says about itself, as plain values.
 *
 * <p>Lives here because reading it means naming GGUF's types, and those stay in the format layer
 * and the loaders. Callers that only want to label a run — the benchmark harness — get strings and
 * a count and never see the header.
 *
 * @param arch {@code general.architecture}, or {@code "unknown"} if the key is absent
 * @param quant {@code general.file_type} under the name llama-bench prints
 * @param paramsB total tensor elements in billions, summed over the tensor infos
 */
public record GgufModelFacts(String arch, String quant, double paramsB) {

    public static GgufModelFacts read(Path path) throws IOException {
        GGUF gguf = GGUF.loadGGUFMetadata(path);
        Object arch = gguf.getMetadata().get("general.architecture");
        Object fileType = gguf.getMetadata().get("general.file_type");
        long params = 0;
        for (var info : gguf.getTensorInfos().values()) {
            long n = 1;
            for (int d : info.dimensions()) {
                n *= d;
            }
            params += n;
        }
        return new GgufModelFacts(
                arch == null ? "unknown" : arch.toString(),
                fileTypeName(fileType instanceof Integer i ? i : -1),
                params / 1e9);
    }

    /** {@code general.file_type} as llama-bench names it. */
    static String fileTypeName(int fileType) {
        return switch (fileType) {
            case 0 -> "F32";
            case 1 -> "F16";
            case 2 -> "Q4_0";
            case 3 -> "Q4_1";
            case 7 -> "Q8_0";
            case 14 -> "Q4_K_S";
            case 15 -> "Q4_K_M";
            case 16 -> "Q5_K_S";
            case 17 -> "Q5_K_M";
            case 18 -> "Q6_K";
            case 32 -> "BF16";
            default -> "type_" + fileType;
        };
    }
}
