package org.beehive.jitllm.tensor.standard;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import jdk.incubator.vector.FloatVector;
import org.junit.Test;

public class QuantizedDotVectorSpeciesTest {

    @Test
    public void onlyImplementedVectorWidthsUseQuantizedVectorKernels() {
        assertTrue(FloatTensor.supportsVectorizedDot(FloatVector.SPECIES_128));
        assertTrue(FloatTensor.supportsVectorizedDot(FloatVector.SPECIES_256));
        assertFalse(FloatTensor.supportsVectorizedDot(FloatVector.SPECIES_512));
    }
}
