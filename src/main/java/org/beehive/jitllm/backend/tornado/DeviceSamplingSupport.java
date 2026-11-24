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
package org.beehive.jitllm.backend.tornado;

import org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy.SamplingResidency;

/**
 * Where greedy sampling on the device ({@code -Djitllm.deviceSample=true}) is implemented: the
 * logits stay on the GPU and only the argmax token id crosses to the host.
 *
 * <p>Each family's plan provider says whether its logits and decode loop do this ({@code
 * TornadoPlanProvider.deviceSampling}). Anywhere else the host needs the whole logits row, so the
 * request is ignored with a warning rather than refused: it is a speed setting, not a different
 * answer.
 */
public final class DeviceSamplingSupport {

    private static final System.Logger LOGGER =
            System.getLogger(DeviceSamplingSupport.class.getName());

    private DeviceSamplingSupport() {}

    /** Whether {@code model}'s GPU plans sample on the device. */
    public static boolean supported(Model model, boolean gpu) {
        return gpu
                && TornadoPlanRegistry.provider(model.architectureId())
                        .map(provider -> provider.deviceSampling(model.weights().dataType()))
                        .orElse(false);
    }

    /** {@code policy}, sampling on the host where the device cannot. */
    public static ExecutionPolicy resolve(Model model, ExecutionPolicy policy, boolean gpu) {
        if (policy.samplingResidency() != SamplingResidency.DEVICE || supported(model, gpu)) {
            return policy;
        }
        LOGGER.log(
                System.Logger.Level.WARNING,
                "device sampling ignored for "
                        + model.architectureId()
                        + " / "
                        + model.weights().dataType()
                        + (gpu ? "" : " on the CPU")
                        + ": sampling on the host");
        return ExecutionPolicy.from(policy).samplingResidency(SamplingResidency.HOST).build();
    }
}
