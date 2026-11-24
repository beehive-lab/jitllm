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
package org.beehive.jitllm.arch.fixture.metrics;

import org.beehive.jitllm.model.AbstractModel;

/**
 * Deliberate Rule 17 violation: a type inside the metrics seam that depends on a layer above it.
 * Exists so {@code DependencyRulesSelfTest} can prove the rule reports something — a rule only ever
 * seen to pass is not a guardrail.
 */
public class ViolatingSeam {

    private final AbstractModel model;

    public ViolatingSeam(AbstractModel model) {
        this.model = model;
    }

    public AbstractModel model() {
        return model;
    }
}
