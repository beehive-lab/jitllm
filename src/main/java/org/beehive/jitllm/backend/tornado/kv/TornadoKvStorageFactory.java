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
