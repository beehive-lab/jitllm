package org.beehive.jllm.api;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Diagnostic, not a gate: what N live GPU sessions cost in device memory, with and without the
 * shared KV pool.
 *
 * <p>{@code -Dprobe.model=. -Dprobe.sessions=N [-Djllm.kv.sharedPool=true]}
 */
public final class SharedKvMemoryProbe {

    public static void main(String[] args) throws Exception {
        Path model = Path.of(System.getProperty("probe.model"));
        int sessions = Integer.getInteger("probe.sessions", 2);
        int holdSeconds = Integer.getInteger("probe.hold", 6);
        System.setProperty("use.tornadovm", "true");

        try (LocalModel loaded =
                LocalModels.load(model, ModelOptions.builder().contextLength(512).build())) {
            TextGenerationModel generator = (TextGenerationModel) loaded;
            List<GenerationSession> open = new ArrayList<>();
            try {
                for (int i = 0; i < sessions; i++) {
                    GenerationSession session = generator.newSession();
                    open.add(session);
                    session.generate(
                            GenerationRequest.builder()
                                    .prompt("Say the number " + i + ".")
                                    .maxNewTokens(4)
                                    .temperature(0.0f)
                                    .build());
                    System.out.println("[probe] session " + (i + 1) + " of " + sessions + " live");
                    System.out.flush();
                }
                System.out.println("[probe] READY");
                System.out.flush();
                Thread.sleep(holdSeconds * 1000L);
            } finally {
                for (GenerationSession session : open) {
                    session.close();
                }
            }
        }
        System.out.println("[probe] DONE");
    }
}
