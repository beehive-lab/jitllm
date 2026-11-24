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
package org.beehive.jitllm.backend.tornado.ops.nativelib;

import java.util.function.BooleanSupplier;

/** Asks whether a native library loads in this process. */
public final class LibraryProbe {

    private LibraryProbe() {}

    /**
     * Whether {@code probe} answers yes. A library whose classes cannot be linked, or whose probe
     * throws, does not load: for example {@code loads(MlxLibraryProvider::isAvailable)} on a
     * machine without MLX is false, not an error.
     */
    public static boolean loads(BooleanSupplier probe) {
        try {
            return probe.getAsBoolean();
        } catch (RuntimeException | LinkageError e) {
            return false;
        }
    }
}
