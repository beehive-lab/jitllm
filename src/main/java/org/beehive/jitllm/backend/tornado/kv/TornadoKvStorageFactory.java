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
package org.beehive.jitllm.backend.tornado.kv;

import org.beehive.jitllm.runtime.kv.KvStorage;
import org.beehive.jitllm.runtime.kv.KvStorageFactory;
import org.beehive.jitllm.runtime.kv.KvStorageRequest;

/**
 * Allocates this backend's {@link TornadoKvStore}. The only place outside tests that names its
 * constructor.
 */
public final class TornadoKvStorageFactory implements KvStorageFactory {

    @Override
    public KvStorage create(KvStorageRequest request) {
        return new TornadoKvStore(
                request.totalBlocks(),
                request.blocksPerSlot(),
                request.maxSlots(),
                request.blockSizeTokens(),
                request.numberOfLayers(),
                request.kvDim(),
                request.fp16KeyValue());
    }
}
