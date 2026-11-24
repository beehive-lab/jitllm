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
package org.beehive.jitllm.runtime.backend;

/**
 * Which implementation actually executed a session — the lowered program, or the legacy plan.
 *
 * <p>Neutral by construction: an enum in {@code runtime.backend}, naming neither TornadoVM nor an
 * implementation package, so the generation layer and the metrics seam can both report it.
 */
public enum ExecutionPath {

    /** The lowered program built from the operation vocabulary. */
    LOWERED,

    /** The legacy hand-built plan. */
    LEGACY;

    /** Lower-case name, for metrics payloads and log lines. */
    public String reportName() {
        return name().toLowerCase(java.util.Locale.ROOT);
    }
}
