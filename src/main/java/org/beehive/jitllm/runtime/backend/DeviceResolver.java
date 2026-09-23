package org.beehive.jllm.runtime.backend;

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
