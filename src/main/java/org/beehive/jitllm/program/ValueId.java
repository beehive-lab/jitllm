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
package org.beehive.jitllm.program;

/**
 * A scalar an invocation supplies, by meaning rather than by buffer.
 *
 * <p>Each is delivered by <b>writing into a persistent control array</b> at a declared offset —
 * never by binding a different array. That is what makes an invocation allocation-free and makes
 * rebinding unrepresentable rather than merely discouraged.
 */
public enum ValueId {

    /** The token being processed. */
    TOKEN,

    /** Its position in the sequence. */
    POSITION,

    /** Which block-table slot this sequence occupies. */
    BLOCK_TABLE_SLOT,

    ACTIVE_COUNT,

    /** The sampler's seed. */
    SAMPLING_SEED,

    /** The sampler's temperature. */
    SAMPLING_TEMPERATURE,

    /** The sampler's nucleus threshold. */
    SAMPLING_TOP_P
}
