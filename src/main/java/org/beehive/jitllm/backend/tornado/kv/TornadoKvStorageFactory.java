package org.beehive.jllm.backend.tornado.kv;

import org.beehive.jllm.runtime.kv.KvStorage;
import org.beehive.jllm.runtime.kv.KvStorageFactory;
import org.beehive.jllm.runtime.kv.KvStorageRequest;

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
