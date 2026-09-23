package org.beehive.jitllm.format;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.ExecutionTarget;
import org.beehive.jitllm.runtime.tensor.TensorDescriptor;
import org.beehive.jitllm.runtime.tensor.TensorLayout;
import org.beehive.jitllm.runtime.tensor.TensorRole;
import org.junit.Test;

public class TensorDescriptorsTest {

    private static GGMLTensorEntry entry(String name, GGMLType type, int... shape) {
        return new GGMLTensorEntry(MemorySegment.NULL, name, type, shape, MemorySegment.NULL);
    }

    @Test
    public void theDescriptorSaysWhatTheTensorBecomesOnTheDevice() {
        TensorDescriptor descriptor =
                TensorDescriptors.describe(
                        entry("blk.0.attn_q.weight", GGMLType.Q4_K, 2048, 2048),
                        ExecutionTarget.GPU);

        assertSame("the device has no K-quant kernel", DataType.Q8_0, descriptor.dataType());
        assertEquals(TensorRole.ATTENTION_QUERY, descriptor.role());
        assertEquals(2048L * 2048, descriptor.elementCount());
        assertTrue(descriptor.layout() instanceof TensorLayout.BlockQuantized);
    }

    @Test
    public void theSameTensorKeepsItsOwnTypeOnTheHost() {
        GGMLTensorEntry entry = entry("blk.0.attn_q.weight", GGMLType.Q4_K, 2048, 2048);
        assertSame(DataType.Q4_K, TensorDescriptors.describeSource(entry).dataType());
        assertSame(
                DataType.Q4_K, TensorDescriptors.describe(entry, ExecutionTarget.CPU).dataType());
    }

    @Test
    public void theLayoutMatchesTheRepresentationItDescribes() {
        assertEquals(
                new TensorLayout.BlockQuantized(32, 34, TensorLayout.ScaleFormat.FP16),
                TensorDescriptors.layoutOf(DataType.Q8_0));
        assertEquals(
                new TensorLayout.BlockQuantized(32, 18, TensorLayout.ScaleFormat.FP16),
                TensorDescriptors.layoutOf(DataType.Q4_0));
        assertEquals(new TensorLayout.Dense(4), TensorDescriptors.layoutOf(DataType.F32));
        assertEquals(new TensorLayout.Dense(2), TensorDescriptors.layoutOf(DataType.F16));
    }

    /**
     * The K-quants' scales are quantized against a super-block, which the layout says and stops at.
     */
    @Test
    public void kQuantScalesAreMarkedHierarchical() {
        TensorLayout layout = TensorDescriptors.layoutOf(DataType.Q6_K);
        assertEquals(
                TensorLayout.ScaleFormat.HIERARCHICAL,
                ((TensorLayout.BlockQuantized) layout).scale());
    }

    /**
     * A layout's byte size must agree with what the format says a tensor of that many elements
     * occupies — this is the number storage is allocated from.
     */
    @Test
    public void theLayoutsByteSizeAgreesWithTheFormatsOwnArithmetic() {
        long elements = 4096L * 11008;
        for (GGMLType fileType :
                new GGMLType[] {GGMLType.F32, GGMLType.F16, GGMLType.Q8_0, GGMLType.Q4_0}) {
            TensorLayout layout = TensorDescriptors.layoutOf(DataTypeMapping.sourceType(fileType));
            long expected = elements / fileType.getBlockSize() * fileType.getTypeSize();
            assertEquals(fileType.toString(), expected, layout.byteSize(elements));
        }
    }

    @Test
    public void everyTensorNameTheLoadersReadHasARole() {
        assertSame(TensorRole.TOKEN_EMBEDDING, TensorDescriptors.roleOf("token_embd.weight"));
        assertSame(TensorRole.OUTPUT, TensorDescriptors.roleOf("output.weight"));
        assertSame(TensorRole.OUTPUT_NORM, TensorDescriptors.roleOf("output_norm.weight"));
        assertSame(TensorRole.ATTENTION_NORM, TensorDescriptors.roleOf("blk.7.attn_norm.weight"));
        assertSame(TensorRole.ATTENTION_QUERY, TensorDescriptors.roleOf("blk.0.attn_q.weight"));
        assertSame(TensorRole.ATTENTION_KEY, TensorDescriptors.roleOf("blk.13.attn_k.weight"));
        assertSame(TensorRole.ATTENTION_VALUE, TensorDescriptors.roleOf("blk.0.attn_v.weight"));
        assertSame(TensorRole.ATTENTION_QKV, TensorDescriptors.roleOf("blk.0.attn_qkv.weight"));
        assertSame(
                TensorRole.ATTENTION_QUERY_NORM,
                TensorDescriptors.roleOf("blk.0.attn_q_norm.weight"));
        assertSame(
                TensorRole.ATTENTION_KEY_NORM,
                TensorDescriptors.roleOf("blk.0.attn_k_norm.weight"));
        assertSame(
                TensorRole.ATTENTION_OUTPUT, TensorDescriptors.roleOf("blk.0.attn_output.weight"));
        assertSame(TensorRole.FFN_NORM, TensorDescriptors.roleOf("blk.0.ffn_norm.weight"));
        assertSame(TensorRole.FFN_GATE, TensorDescriptors.roleOf("blk.0.ffn_gate.weight"));
        assertSame(TensorRole.FFN_UP, TensorDescriptors.roleOf("blk.0.ffn_up.weight"));
        assertSame(TensorRole.FFN_DOWN, TensorDescriptors.roleOf("blk.0.ffn_down.weight"));
    }

    /** An unknown name is OTHER, never a guess — biases are the case that exists today. */
    @Test
    public void anUnrecognizedNameIsNotGuessedAt() {
        assertSame(TensorRole.OTHER, TensorDescriptors.roleOf("blk.0.attn_q.bias"));
        assertSame(TensorRole.OTHER, TensorDescriptors.roleOf("something.new.weight"));
        assertSame(TensorRole.OTHER, TensorDescriptors.roleOf(null));
    }

    /** The layer index is not part of the role: a role identifies what, not where. */
    @Test
    public void theLayerNumberDoesNotChangeTheRole() {
        assertSame(
                TensorDescriptors.roleOf("blk.0.ffn_up.weight"),
                TensorDescriptors.roleOf("blk.31.ffn_up.weight"));
    }
}
