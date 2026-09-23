package org.beehive.jllm.arch.fixture.spi;

import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;

/** Never referenced by production code. */
public class ViolatingNeutralSpi {

    /** A neutral contract that hands back an implementation type defeats the whole arrangement. */
    public TornadoVMMasterPlan leakTheImplementation() {
        return null;
    }
}
