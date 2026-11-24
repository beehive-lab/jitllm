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

/** What a runtime can be asked about the key/value capacity it owns. */
public interface CapacityQuery {

    /** The pool's totals and what is left of it, at this moment. */
    KvCapacity capacity();

    /**
     * The most tokens one slot can ever hold — the ceiling a request is rejected against when it
     * could not fit even in an empty pool.
     *
     * <p>Distinct from {@link #capacity()} because it is a property of how the pool is shaped
     * rather than of how full it is: a request larger than this is not waiting for room, it is
     * impossible, and telling those two apart is the difference between a queue and a hang.
     */
    int tokensPerSlot();
}
