package org.beehive.jitllm.inference.state;

import org.beehive.jitllm.model.kimilinear.KimiLinearConfiguration;

/** The device arrays of a {@code kimi-linear} session. */
final class KimiLinearDeviceWorkspace {

    private KimiLinearDeviceWorkspace() {}

    static void allocate(KimiLinearState state, KimiLinearConfiguration config) {
        throw new UnsupportedOperationException("kimi-linear does not run on the accelerator yet");
    }
}
