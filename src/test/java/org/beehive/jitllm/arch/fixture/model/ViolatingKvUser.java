package org.beehive.jitllm.arch.fixture.model;

/** Deliberate Rule 7 violator: a model-layer type reaching a KV storage type. */
public class ViolatingKvUser {
    public KvCacheBlockPool pool;
}
