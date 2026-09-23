package org.beehive.jllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.nio.file.Path;
import java.util.Optional;
import java.util.Set;
import org.beehive.jllm.runtime.tensor.DataType;
import org.junit.Test;

public class ModelInfoTest {

    private static ModelInfo info(Set<DataType> weights, DataType compute) {
        return new ModelInfo("Llama-3.2-1B", "llama", 512, Path.of("m.gguf"), weights, compute);
    }

    /**
     * The acceptance in one test: a Q6_K file loaded for the device reports Q8_0, because that is
     * what its weights were materialized as. Reporting Q6_K would describe the file and mislead
     * about what runs.
     */
    @Test
    public void aKQuantFileReportsWhatItWasMaterializedAs() {
        ModelInfo materialized = info(Set.of(DataType.Q8_0), DataType.Q8_0);
        assertEquals(Optional.of(DataType.Q8_0), materialized.weightType());
        assertEquals(DataType.Q8_0, materialized.computeType());
    }

    @Test
    public void anFp16ModelComputesInFp16() {
        assertEquals(DataType.F16, info(Set.of(DataType.F16), DataType.F16).computeType());
    }

    /**
     * A mixed weight set has no single answer, and saying so is the honest option: picking one
     * value would be choosing which tensors to describe without mentioning the choice.
     */
    @Test
    public void aMixedWeightSetHasNoSingleWeightType() {
        ModelInfo mixed = info(Set.of(DataType.Q8_0, DataType.F16), DataType.Q8_0);
        assertEquals(Optional.empty(), mixed.weightType());
        assertEquals(Set.of(DataType.Q8_0, DataType.F16), mixed.weightTypes());
        assertTrue(mixed.toString(), mixed.toString().contains("mixed"));
    }

    @Test
    public void aModelWithNoWeightRepresentationIsNonsense() {
        assertThrows(IllegalArgumentException.class, () -> info(Set.of(), DataType.Q8_0));
    }

    @Test
    public void whatItReportsCannotBeChangedAfterwards() {
        ModelInfo model = info(Set.of(DataType.Q8_0), DataType.Q8_0);
        assertThrows(
                UnsupportedOperationException.class, () -> model.weightTypes().add(DataType.F16));
    }
}
