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
package org.beehive.jitllm.backend.tornado.ops;

import java.util.List;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import uk.ac.manchester.tornado.api.KernelContext;

/**
 * One way to run program operations: with JIT kernels, or with a vendor library such as cuBLAS or
 * MLX.
 *
 * <p>Its fusion table says which operation sequences it runs as one step. To let a vendor library
 * run more operations, add one class that implements this interface and list it in the plan's
 * candidates.
 */
public interface Implementation {

    /** The vendor library used, for example {@code "cuBLAS"}, or {@code "none"}. */
    String library();

    /** The operation sequences it runs as one step, longest first. */
    List<Fusion> fusions();

    /** Whether it can run {@code step} on {@code target}. */
    boolean accepts(Step step, Target target);

    /** The device memory {@link #create} will allocate for {@code steps}, computed up front. */
    default BatchPrefillWorkspace needs(List<Step> steps, Target target) {
        return BatchPrefillWorkspace.withoutResidencyChoice(0);
    }

    /** Allocates what {@code steps} need and returns the object that adds their tasks. */
    Instance create(
            List<Step> steps, Target target, KernelContext context, WeightPrepacking prepacking);

    /**
     * {@link #create} for the narrow family ({@code PrefillFamily}), whose {@code steps} are the
     * primary's over fewer rows: an implementation that allocates takes {@code primary}'s buffers,
     * the instance it created for the primary. One that allocates nothing creates as usual.
     */
    default Instance narrowed(
            Instance primary,
            List<Step> steps,
            Target target,
            KernelContext context,
            WeightPrepacking prepacking) {
        return create(steps, target, context, prepacking);
    }
}
