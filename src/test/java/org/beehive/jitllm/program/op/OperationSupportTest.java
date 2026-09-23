package org.beehive.jllm.program.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.EnumSet;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.runtime.tensor.ExecutionTarget;
import org.junit.Test;

public class OperationSupportTest {

    private static final OperandRef.Weight W =
            new OperandRef.Weight("w", org.beehive.jllm.runtime.tensor.TensorRole.OUTPUT);
    private static final OperandRef A = new OperandRef.Activation("a");
    private static final OperandRef B = new OperandRef.Activation("b");

    /** The K-quants execute on the host, decoded inside the dot product. */
    @Test
    public void theHostExecutesKquantsDirectly() {
        for (DataType kQuant : Set.of(DataType.Q4_0, DataType.Q4_K, DataType.Q5_K, DataType.Q6_K)) {
            assertTrue(
                    kQuant + " is decoded in the CPU dot product and must be supported there",
                    OperationSupport.supports(OperationKind.MAT_VEC, kQuant, ExecutionTarget.CPU));
        }
    }

    /**
     * The accelerator reads quantized weights in the file's own layout, for the operations that
     * have kernels for them.
     *
     * <p>This replaces a test asserting the opposite. The engine used to materialize every
     * block-encoded representation as {@code Q8_0} before it reached a device, and that test
     * enforced the consequence — that no such representation could be claimed on the GPU. Both are
     * gone. Decoding a block inside the arithmetic is what a device kernel does as much as a host
     * dot product, and materializing instead nearly doubles a 4-bit model's device footprint.
     *
     * <p>What is still true, and what this asserts, is narrower: support is per operation. A
     * matrix-vector product reads all six; matrix-matrix has tensor-core kernels for two.
     */
    @Test
    public void theAcceleratorReadsQuantizedWeightsInTheirOwnLayout() {
        for (DataType quantized :
                Set.of(
                        DataType.Q4_0,
                        DataType.Q4_1,
                        DataType.Q4_K,
                        DataType.Q5_K,
                        DataType.Q6_K,
                        DataType.Q8_0)) {
            assertTrue(
                    quantized + " is decoded inside the device dot product and must be supported",
                    OperationSupport.supports(
                            OperationKind.MAT_VEC, quantized, ExecutionTarget.GPU));
            assertTrue(
                    quantized + " must also be readable by the vocabulary projection",
                    OperationSupport.supports(
                            OperationKind.VOCAB_PROJECTION, quantized, ExecutionTarget.GPU));
        }
        // Per operation, not per backend: the batched tensor-core path has two representations.
        assertFalse(
                "MAT_MUL has no quantized tensor-core kernel beyond Q8_0",
                OperationSupport.supports(
                        OperationKind.MAT_MUL, DataType.Q5_K, ExecutionTarget.GPU));
    }

    /**
     * {@code BF16} is narrowed to {@code F16} for the device and read directly on the host, so it
     * belongs to exactly one of the two columns.
     */
    @Test
    public void bf16IsHostOnlyUntilThereAreKernelsForIt() {
        assertTrue(
                OperationSupport.supports(
                        OperationKind.MAT_VEC, DataType.BF16, ExecutionTarget.CPU));
        assertFalse(
                OperationSupport.supports(
                        OperationKind.MAT_VEC, DataType.BF16, ExecutionTarget.GPU));
        assertEquals(
                "the narrowing the loader performs must be the one DataType states",
                DataType.F16,
                DataType.BF16.narrowedFallback());
    }

    /**
     * The host has no matrix-matrix product: its batched prefill holds an array of per-row tensors
     * and multiplies them one row at a time. An empty set says so; a populated one would claim an
     * implementation that does not exist.
     */
    @Test
    public void theHostDeclaresNoMatrixMatrixProduct() {
        assertTrue(
                OperationSupport.supported(OperationKind.MAT_MUL, ExecutionTarget.CPU).isEmpty());
        assertFalse(
                OperationSupport.supported(OperationKind.MAT_MUL, ExecutionTarget.GPU).isEmpty());
    }

    /** Every kind is declared for every target — an undeclared kind would be an unreadable gap. */
    @Test
    public void theTableIsTotal() {
        for (OperationKind kind : OperationKind.values()) {
            for (ExecutionTarget target : ExecutionTarget.values()) {
                Set<DataType> supported = OperationSupport.supported(kind, target);
                assertTrue(
                        "supported(" + kind + ", " + target + ") must not be null",
                        supported != null);
            }
        }
    }

    /** Support sets are immutable: a caller must not be able to grant itself a representation. */
    @Test
    public void supportSetsAreImmutable() {
        for (OperationKind kind : OperationKind.values()) {
            for (ExecutionTarget target : ExecutionTarget.values()) {
                Set<DataType> supported = OperationSupport.supported(kind, target);
                try {
                    supported.add(DataType.F32);
                    fail("supported(" + kind + ", " + target + ") must be immutable");
                } catch (UnsupportedOperationException expected) {
                    // the contract
                }
            }
        }
    }

    /**
     * An unsupported pair is refused before invocation, naming the operation and the dtype.
     *
     * <p>The example is a quantized matrix-<i>matrix</i> product, which is genuinely unsupported:
     * the batched tensor-core kernels exist for F16 and Q8_0 only. It used to be a quantized
     * matrix-vector product, which is no longer unsupported at all.
     */
    @Test
    public void anUnsupportedPairIsRefusedByName() {
        Operation kQuantBatchedOnTheDevice = new MatMul(W, A, B, 4096, 4096, 8, DataType.Q5_K);
        try {
            OperationSupport.require(kQuantBatchedOnTheDevice, ExecutionTarget.GPU);
            fail("Q5_K has no tensor-core kernel and must be refused");
        } catch (UnsupportedOperationException e) {
            String message = e.getMessage();
            assertTrue(
                    "the message must name the operation: " + message, message.contains("MAT_MUL"));
            assertTrue(
                    "the message must name the representation: " + message,
                    message.contains("Q5_K"));
            assertTrue("the message must name the target: " + message, message.contains("GPU"));
        }
    }

    /** An operation the target does not implement at all says so, rather than listing nothing. */
    @Test
    public void anUnimplementedOperationSaysSo() {
        Operation matMulOnTheHost = new MatMul(W, A, B, 4096, 4096, 8, DataType.F32);
        try {
            OperationSupport.require(matMulOnTheHost, ExecutionTarget.CPU);
            fail("the host has no matrix-matrix product");
        } catch (UnsupportedOperationException e) {
            assertTrue(
                    "the message must distinguish 'no kernel for this dtype' from 'no"
                            + " implementation at all': "
                            + e.getMessage(),
                    e.getMessage().contains("does not implement"));
        }
    }

    /** A supported pair passes silently — the check is a guard, not a ceremony. */
    @Test
    public void aSupportedPairPasses() {
        OperationSupport.require(
                new MatVec(W, A, B, 4096, 4096, DataType.Q8_0), ExecutionTarget.GPU);
        OperationSupport.require(
                new MatVec(W, A, B, 4096, 4096, DataType.Q4_K), ExecutionTarget.CPU);
        OperationSupport.require(
                new RoPE(A, B, Optional.empty(), 128, 500000f, RopeLayout.NEOX_HALF, DataType.F32),
                ExecutionTarget.CPU);
    }

    @Test
    public void everyDeclaredRepresentationIsAKnownDataType() {
        Set<DataType> declared = OperationSupport.everySupportedDataType();
        Set<String> missing = new TreeSet<>();
        for (DataType dataType : EnumSet.allOf(DataType.class)) {
            if (!declared.contains(dataType)) {
                missing.add(dataType.name());
            }
        }
        assertTrue(
                "DataType values no target runs any operation at: "
                        + missing
                        + " — either a kernel is missing or the value was added ahead of one",
                missing.isEmpty());
    }
}
