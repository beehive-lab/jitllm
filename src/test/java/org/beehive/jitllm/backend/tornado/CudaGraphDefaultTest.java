package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** CUDA graphs: on by default on CUDA, off when declined, never on another backend. */
public class CudaGraphDefaultTest {

    @Test
    public void onByDefaultOnCuda() {
        assertTrue(CudaGraphDefault.enabled(null, true));
        assertTrue(CudaGraphDefault.enabled("true", true));
    }

    @Test
    public void offWhenDeclined() {
        assertFalse(CudaGraphDefault.enabled("false", true));
    }

    @Test
    public void neverOnWithoutCuda() {
        assertFalse(CudaGraphDefault.enabled(null, false));
        assertFalse(CudaGraphDefault.enabled("true", false));
    }
}
