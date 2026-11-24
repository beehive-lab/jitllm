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
 * What a program-fixed device array is for.
 *
 * <p>Every device array bound into a captured graph has one of these — weights and key/value
 * storage as much as workspace, control and result arrays. Inputs and outputs are not exempt: an
 * output array is a device array in a captured graph, and its identity is as fixed as a weight's
 */
public enum BindingRole {

    /** Model weights. */
    WEIGHT,

    /** Retained key/value storage. */
    KV_POOL,

    /** The block table mapping logical positions to pool blocks. */
    BLOCK_TABLE,

    /** Activations and scratch that live for one invocation but are allocated once. */
    WORKSPACE,

    /** Staging buffers a batched phase fills before execution. */
    STAGING,

    /** Persistent arrays invocation values are written into. */
    CONTROL,

    /** Arrays host-visible results are read out of. */
    RESULT
}
