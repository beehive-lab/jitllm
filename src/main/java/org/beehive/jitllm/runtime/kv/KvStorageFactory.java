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
package org.beehive.jitllm.runtime.kv;

/**
 * Allocates the device storage a {@link KvCacheManager} leases from — implemented by a backend,
 * discovered through {@code ServiceLoader}.
 *
 * <p><b>Allocation only.</b> No per-token operation, and nothing that resolves a lease — a lease
 * resolves once to a view through the binder, and the hot path is the in-kernel block-table walk
 */
public interface KvStorageFactory {

    /**
     * Allocates a pool.
     *
     * @throws RuntimeException or {@link OutOfMemoryError} if the device cannot fit it — a capacity
     *     failure, which callers may treat as a reason to fall back to per-session storage. A
     *     <i>missing</i> factory is not this: it is a configuration error, and it is raised by
     *     {@link KvStorageFactories} before this method is ever reached
     */
    KvStorage create(KvStorageRequest request);
}
