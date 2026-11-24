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
package org.beehive.jitllm.golden;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.beehive.jitllm.backend.tornado.plan.ForwardPlanFactory;
import org.beehive.jitllm.backend.tornado.plan.SingleTokenForwardPlan;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

/**
 * Run per backend, e.g. {@code -Dtornado.print.kernel=True} is not used — {@code withPrintKernel()}
 * is set here, before compilation, which is the only point at which the dump happens.
 */
public final class PagedKernelInliningCheck {

    public static void main(String[] args) throws Exception {
        Path modelPath = Path.of(System.getProperty("check.model"));
        Path outDir = Path.of(System.getProperty("check.out", "kernel-dump"));
        Files.createDirectories(outDir);

        Model model = ModelLoader.loadModel(modelPath, 512, true, true);
        State state = model.createNewState();
        System.out.println(
                "[check] blockCfg=" + state.kvBlockCfg + " blockStride=" + state.kvBlockStride);

        SingleTokenForwardPlan forwardPlan =
                ForwardPlanFactory.createSingleToken(model.weights().dataType(), state, model);

        ProgramIdentity.SourceRecorder recorder = ProgramIdentity.SourceRecorder.install();
        List<String> sources;
        try {
            List<ImmutableTaskGraph> graphs = forwardPlan.getImmutableTaskGraphs();
            TornadoExecutionPlan plan =
                    new TornadoExecutionPlan(graphs.toArray(new ImmutableTaskGraph[0]));
            plan.withProfiler(ProfilerMode.SILENT);
            plan.withPrintKernel();
            plan.withPreCompilation();
            sources = recorder.sources();
        } finally {
            recorder.uninstall();
        }

        System.out.println("[check] compiled modules: " + sources.size());
        int attentionSeen = 0;
        for (String source : sources) {
            List<String> entries = ProgramIdentity.entryPoints(source);
            String name = entries.isEmpty() ? "unnamed" : entries.get(0);
            Files.writeString(outDir.resolve(name + ".txt"), source);
            if (!name.contains("attention") && !name.contains("rope")) {
                continue;
            }
            attentionSeen++;
            boolean callsHelper = source.contains("KvBlockAddress") || source.contains("offset(");
            boolean hasModulo = source.contains("%") || source.contains("rem");
            System.out.println(
                    "[check] "
                            + name
                            + " bytes="
                            + source.length()
                            + " residualCall="
                            + callsHelper
                            + " hasRemainder="
                            + hasModulo
                            + " entryPoints="
                            + entries.size());
        }
        System.out.println(
                "[check] KV kernels inspected: "
                        + attentionSeen
                        + " (dump in "
                        + outDir.toAbsolutePath()
                        + ")");
    }
}
