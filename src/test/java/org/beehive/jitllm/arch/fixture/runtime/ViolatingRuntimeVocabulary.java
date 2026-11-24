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
package org.beehive.jitllm.arch.fixture.runtime;

import org.beehive.jitllm.format.GGMLType;

/**
 * Deliberate Rule 4 violation of the <i>format</i> kind: a runtime-layer type naming a file-format
 * type. The backend half of the rule is covered by {@code fixture.model.ViolatingModel}, which
 * names TornadoVM types; without this one, the format half would never be seen to fail.
 */
public class ViolatingRuntimeVocabulary {

    private final GGMLType fileType;

    public ViolatingRuntimeVocabulary(GGMLType fileType) {
        this.fileType = fileType;
    }

    public GGMLType fileType() {
        return fileType;
    }
}
