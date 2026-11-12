package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.ops.Region;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.architecture.GraniteProgramDescription;
import org.beehive.jitllm.model.architecture.LlamaProgramDescription;
import org.beehive.jitllm.model.architecture.Phi3ProgramDescription;
import org.beehive.jitllm.model.architecture.Qwen2ProgramDescription;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.model.phi3.Phi3Configuration;
import org.beehive.jitllm.model.qwen2.Qwen2Configuration;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.ProgramComponent;
import org.beehive.jitllm.program.op.Attention;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.program.op.MatVec;
import org.beehive.jitllm.program.op.Operation;
import org.beehive.jitllm.program.op.RmsNorm;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.TensorRole;
import org.junit.Test;

/**
 * Each Llama-shaped family's program description and its batched prefill regions ({@code
 * LlamaPrefillRegions}) say the same thing: the same projection roles and widths in the same order,
 * the same norms, and the family's own attention. Llama, Qwen2 (whose biases stay layer tasks),
 * Phi-3 (fused q|k|v and gate|up) and Granite (scores multiplied by its attention multiplier).
 *
 * <p>Example: before one attention definition per family, the regions stored the scale as
 * 1/sqrt(head) for every family, Granite included, where the descriptions divide by sqrt(head) or,
 * for Granite, multiply by the multiplier.
 */
public class LlamaFamilyLayerSemanticsTest {

    private static final int DIM = 96;
    private static final int HIDDEN = 160;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HEAD = DIM / HEADS;
    private static final int TOKENS = 8;
    private static final int ROWS = 64;
    private static final float EPS = 1e-5f;
    private static final float GRANITE_SCALE = 0.0078125f;

    /** Role, output width and input width of one projection. */
    private record Projection(TensorRole role, int outputs, int inputs) {}

    @Test
    public void llama() {
        LlamaConfiguration config =
                new LlamaConfiguration("FP16", DIM, HIDDEN, 2, HEADS, KV_HEADS, 48, 32, EPS, 5e5f);
        assertFamily(
                config,
                LlamaProgramDescription.build(config, DataType.F16, DataType.F16, false, false),
                false);
    }

    @Test
    public void qwen2() {
        Qwen2Configuration config =
                new Qwen2Configuration(
                        "FP16", DIM, HIDDEN, 2, HEADS, KV_HEADS, HEAD, HEAD, 48, 32, 32, false, EPS,
                        1e6f);
        assertFamily(
                config,
                Qwen2ProgramDescription.build(config, DataType.F16, DataType.F16, false, false),
                false);
    }

    @Test
    public void phi3() {
        Phi3Configuration config =
                new Phi3Configuration("FP16", DIM, HIDDEN, 2, HEADS, KV_HEADS, 48, 32, EPS, 1e4f);
        assertFamily(
                config,
                Phi3ProgramDescription.build(config, DataType.F16, DataType.F16, false, false),
                true);
    }

    @Test
    public void granite() {
        GraniteConfiguration config =
                new GraniteConfiguration(
                        "FP16",
                        DIM,
                        HIDDEN,
                        2,
                        HEADS,
                        KV_HEADS,
                        48,
                        32,
                        EPS,
                        1e4f,
                        12.0f,
                        0.22f,
                        GRANITE_SCALE,
                        16.0f,
                        true);
        Attention attention =
                assertFamily(
                        config,
                        GraniteProgramDescription.build(
                                config, DataType.F16, DataType.F16, false, false),
                        false);
        assertEquals(Attention.ScoreScaling.MULTIPLY, attention.scoreScaling());
        assertEquals(GRANITE_SCALE, attention.scale(), 0f);
    }

    /** The checks every family shares; returns the description's attention. */
    private static Attention assertFamily(
            Configuration config, InferenceProgram program, boolean fused) {
        List<Operation> description = layerOps(program);
        List<Operation> regions = new ArrayList<>();
        for (Region region :
                LlamaPrefillRegions.of(
                                config,
                                TOKENS,
                                ROWS,
                                fused,
                                true,
                                LlamaFamilyLayerSemanticsTest::type)
                        .get(0)
                        .all()) {
            regions.addAll(region.ops());
        }

        List<Projection> expected =
                fused
                        ? List.of(
                                new Projection(
                                        TensorRole.ATTENTION_QKV, DIM + 2 * KV_HEADS * HEAD, DIM),
                                new Projection(TensorRole.ATTENTION_OUTPUT, DIM, DIM),
                                new Projection(TensorRole.FFN_GATE_UP, 2 * HIDDEN, DIM),
                                new Projection(TensorRole.FFN_DOWN, DIM, HIDDEN))
                        : List.of(
                                new Projection(TensorRole.ATTENTION_QUERY, DIM, DIM),
                                new Projection(TensorRole.ATTENTION_KEY, KV_HEADS * HEAD, DIM),
                                new Projection(TensorRole.ATTENTION_VALUE, KV_HEADS * HEAD, DIM),
                                new Projection(TensorRole.ATTENTION_OUTPUT, DIM, DIM),
                                new Projection(TensorRole.FFN_GATE, HIDDEN, DIM),
                                new Projection(TensorRole.FFN_UP, HIDDEN, DIM),
                                new Projection(TensorRole.FFN_DOWN, DIM, HIDDEN));
        List<Projection> programProjections = new ArrayList<>();
        List<RmsNorm> programNorms = new ArrayList<>();
        Attention programAttention = null;
        for (Operation op : description) {
            if (op instanceof MatVec m) {
                programProjections.add(new Projection(m.weight().role(), m.rows(), m.columns()));
            } else if (op instanceof RmsNorm norm) {
                programNorms.add(norm);
            } else if (op instanceof Attention a) {
                programAttention = a;
            }
        }
        List<Projection> regionProjections = new ArrayList<>();
        List<RmsNorm> regionNorms = new ArrayList<>();
        for (Operation op : regions) {
            if (op instanceof MatMul m) {
                regionProjections.add(new Projection(m.weight().role(), m.rows(), m.columns()));
                assertEquals(ROWS, m.batchSize());
                assertEquals(type(m.weight().role(), 0), m.dataType());
            } else if (op instanceof RmsNorm norm) {
                regionNorms.add(norm);
            }
        }
        assertEquals(expected, programProjections);
        assertEquals(expected, regionProjections);

        assertEquals(2, programNorms.size());
        assertEquals(2, regionNorms.size());
        for (int i = 0; i < 2; i++) {
            assertEquals(EPS, programNorms.get(i).epsilon(), 0f);
            assertEquals(EPS, regionNorms.get(i).epsilon(), 0f);
            assertEquals(
                    programNorms.get(i).weight().get().role(),
                    regionNorms.get(i).weight().get().role());
        }

        Attention regionAttention =
                (Attention)
                        LlamaPrefillRegions.attention(config, TOKENS, fused).get(0).ops().get(0);
        assertEquals(HEADS, programAttention.heads());
        assertEquals(KV_HEADS, programAttention.keyValueHeads());
        assertEquals(HEAD, programAttention.headDimension());
        assertFalse(programAttention.window().isPresent());
        assertEquals(programAttention.heads(), regionAttention.heads());
        assertEquals(programAttention.keyValueHeads(), regionAttention.keyValueHeads());
        assertEquals(programAttention.headDimension(), regionAttention.headDimension());
        assertEquals(programAttention.scale(), regionAttention.scale(), 0f);
        assertEquals(programAttention.scoreScaling(), regionAttention.scoreScaling());
        assertEquals(programAttention.window(), regionAttention.window());
        if (programAttention.scoreScaling() == Attention.ScoreScaling.DIVIDE) {
            // The divisor the lowerings check, not its reciprocal.
            assertEquals((float) Math.sqrt(HEAD), programAttention.scale(), 0f);
        }
        assertTrue(regionAttention.dataType() == DataType.F32);
        return programAttention;
    }

    /** Mixed per tensor, as K-quant files are: attention in F16, the feed-forward in Q8_0. */
    private static DataType type(TensorRole role, int layer) {
        return role.name().startsWith("FFN") ? DataType.Q8_0 : DataType.F16;
    }

    /** The operations of the program's first layer component, in order. */
    private static List<Operation> layerOps(InferenceProgram program) {
        for (ProgramComponent component : program.components()) {
            if (component instanceof ProgramComponent.Composite layer) {
                List<Operation> ops = new ArrayList<>();
                for (ProgramComponent child : layer.children()) {
                    ops.add(((ProgramComponent.Leaf) child).operation());
                }
                return ops;
            }
        }
        throw new AssertionError("no layer component");
    }
}
