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
 * Resolves the device this process actually runs an accelerator on — implemented by a backend,
 * discovered through {@code ServiceLoader}, exactly the shape {@code runtime.kv.KvStorageFactory}
 * already uses for the equivalent allocation-side question.
 *
 * <p><b>Not a selection mechanism.</b> This answers "what did the process resolve", never "resolve
 * me a particular one" — there is no parameter, and calling it twice must answer the same identity
 * for the life of the process, the same stability {@code TornadoDevices.current()} already gives.
 */
public interface DeviceResolver {

    /** The device this process's backend has already resolved. Never {@code null}. */
    Device resolve();
}
