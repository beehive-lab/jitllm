/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package org.beehive.jitllm.api;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Diagnostic, not a gate: what N live GPU sessions cost in device memory, with and without the
 * shared KV pool.
 *
 * <p>{@code -Dprobe.model=. -Dprobe.sessions=N [-Djitllm.kv.sharedPool=true]}
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
