package org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.layers.Qwen3PrefillRegions;
import org.beehive.jitllm.backend.tornado.ops.Region;
import org.beehive.jitllm.model.architecture.Qwen3ProgramDescription;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.ProgramComponent;
import org.beehive.jitllm.program.op.Attention;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.program.op.MatVec;
import org.beehive.jitllm.program.op.Operation;
import org.beehive.jitllm.program.op.OperationKind;
import org.beehive.jitllm.program.op.RmsNorm;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.TensorRole;
import org.junit.Test;

/**
 * The three descriptions of a Qwen3 layer say the same thing: the single-token program ({@code
 * Qwen3ProgramDescription}), the batched prefill regions ({@code Qwen3PrefillRegions}) and the
 * tensor-core prefill's fused regions ({@code Qwen3FP16LayersBatchPrefillMMA}). Same projection
 * roles and widths in the same order, same norms, same residual and SwiGLU wiring, same attention.
 *
 * <p>The configuration keeps dim (96), the query width (4 x 32 = 128), the key/value width (2 x 32
 * = 64) and hidden (160) all different, so a swapped width fails here. Example: before one
 * attention definition, the regions stored the scale as 1/sqrt(32) where the program and its
 * lowering use the divisor sqrt(32).
 */
public class Qwen3LayerSemanticsTest {

    private static final int DIM = 96;
    private static final int HIDDEN = 160;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HEAD = 32;
    private static final int TOKENS = 8;
    private static final int ROWS = 16;
    private static final float EPS = 1e-6f;

    private static final Qwen3Configuration CONFIG =
            new Qwen3Configuration(
                    "F16",
                    DIM,
                    HIDDEN,
                    2,
                    HEADS,
                    KV_HEADS,
                    HEAD,
                    HEAD,
                    64,
                    32,
                    32,
                    false,
                    EPS,
                    1_000_000f);

    /** Role, output width and input width of one projection. */
    private record Projection(TensorRole role, int outputs, int inputs) {}

    @Test
    public void projectionsHaveTheSameRolesAndWidthsInTheSameOrder() {
        List<Projection> expected =
                List.of(
                        new Projection(TensorRole.ATTENTION_QUERY, HEADS * HEAD, DIM),
                        new Projection(TensorRole.ATTENTION_KEY, KV_HEADS * HEAD, DIM),
                        new Projection(TensorRole.ATTENTION_VALUE, KV_HEADS * HEAD, DIM),
                        new Projection(TensorRole.ATTENTION_OUTPUT, DIM, HEADS * HEAD),
                        new Projection(TensorRole.FFN_GATE, HIDDEN, DIM),
                        new Projection(TensorRole.FFN_UP, HIDDEN, DIM),
                        new Projection(TensorRole.FFN_DOWN, DIM, HIDDEN));
        List<Projection> program = new ArrayList<>();
        for (Operation op : programLayer()) {
            if (op instanceof MatVec m) {
                program.add(new Projection(m.weight().role(), m.rows(), m.columns()));
            }
        }
        List<Projection> regions = new ArrayList<>();
        for (Operation op : regionLayer()) {
            if (op instanceof MatMul m) {
                regions.add(new Projection(m.weight().role(), m.rows(), m.columns()));
                assertEquals(ROWS, m.batchSize());
            }
        }
        assertEquals(expected, program);
        assertEquals(expected, regions);
    }

    @Test
    public void theFusedRegionsAreTheSumsOfTheirParts() {
        Qwen3FP16LayersBatchPrefillMMA.LayerRegions layer =
                Qwen3FP16LayersBatchPrefillMMA.regions(CONFIG, TOKENS).get(0);
        assertEquals(
                new Projection(TensorRole.ATTENTION_QKV, (HEADS + 2 * KV_HEADS) * HEAD, DIM),
                projection(layer.qkv()));
        assertEquals(
                new Projection(TensorRole.ATTENTION_OUTPUT, DIM, HEADS * HEAD),
                projection(layer.output()));
        assertEquals(
                new Projection(TensorRole.FFN_GATE_UP, 2 * HIDDEN, DIM),
                projection(layer.gateUp()));
        assertEquals(new Projection(TensorRole.FFN_DOWN, DIM, HIDDEN), projection(layer.down()));
    }

    @Test
    public void everyDescriptionHasTheSameAttention() {
        Attention program = null;
        for (Operation op : programLayer()) {
            if (op instanceof Attention a) {
                program = a;
            }
        }
        Attention regions =
                (Attention) Qwen3PrefillRegions.attention(CONFIG, TOKENS).get(0).ops().get(0);
        Attention fused =
                (Attention)
                        Qwen3FP16LayersBatchPrefillMMA.regions(CONFIG, TOKENS)
                                .get(0)
                                .attention()
                                .ops()
                                .get(0);
        for (Attention attention : List.of(regions, fused)) {
            assertEquals(program.heads(), attention.heads());
            assertEquals(program.keyValueHeads(), attention.keyValueHeads());
            assertEquals(program.headDimension(), attention.headDimension());
            assertEquals(program.scale(), attention.scale(), 0f);
            assertEquals(program.scoreScaling(), attention.scoreScaling());
            assertEquals(program.window(), attention.window());
        }
        assertEquals(HEADS, program.heads());
        assertEquals(KV_HEADS, program.keyValueHeads());
        assertEquals(HEAD, program.headDimension());
        // The divisor the lowering checks, not its reciprocal.
        assertEquals((float) Math.sqrt(HEAD), program.scale(), 0f);
        assertEquals(Attention.ScoreScaling.DIVIDE, program.scoreScaling());
        assertFalse(program.window().isPresent());
    }

    @Test
    public void normsResidualsAndSwiGluAreWiredAlike() {
        // The program's per-head q/k norms, RoPE, key/value append and attention are the layer
        // classes' own tasks around the regions; the rest must match one for one.
        List<OperationKind> program = new ArrayList<>();
        List<RmsNorm> programNorms = new ArrayList<>();
        for (Operation op : programLayer()) {
            if (op instanceof RmsNorm norm && norm.isGrouped()) {
                continue;
            }
            switch (op.kind()) {
                case ROPE, KV_APPEND, ATTENTION -> {
                    continue;
                }
                case MAT_VEC -> program.add(OperationKind.MAT_MUL);
                default -> program.add(op.kind());
            }
            if (op instanceof RmsNorm norm) {
                programNorms.add(norm);
            }
        }
        List<OperationKind> regions = new ArrayList<>();
        List<RmsNorm> regionNorms = new ArrayList<>();
        for (Operation op : regionLayer()) {
            regions.add(op.kind());
            if (op instanceof RmsNorm norm) {
                regionNorms.add(norm);
            }
        }
        assertEquals(program, regions);
        assertEquals(2, programNorms.size());
        for (int i = 0; i < programNorms.size(); i++) {
            RmsNorm a = programNorms.get(i);
            RmsNorm b = regionNorms.get(i);
            assertEquals(EPS, a.epsilon(), 0f);
            assertEquals(a.epsilon(), b.epsilon(), 0f);
            assertEquals(a.weight().get().role(), b.weight().get().role());
        }
    }

    @Test
    public void eachProjectionKeepsItsOwnTensorType() {
        for (Operation op : regionLayer()) {
            if (op instanceof MatMul m) {
                assertEquals(type(m.weight().role(), 0), m.dataType());
            }
        }
    }

    /** Mixed per tensor, as K-quant files are: attention in F16, the feed-forward in Q8_0. */
    private static DataType type(TensorRole role, int layer) {
        return role.name().startsWith("FFN") ? DataType.Q8_0 : DataType.F16;
    }

    private static List<Operation> programLayer() {
        InferenceProgram program =
                Qwen3ProgramDescription.build(CONFIG, DataType.F16, DataType.F16, false, false);
        List<Operation> ops = new ArrayList<>();
        for (ProgramComponent child :
                ((ProgramComponent.Composite) program.components().get(1)).children()) {
            ops.add(((ProgramComponent.Leaf) child).operation());
        }
        return ops;
    }

    /** Layer 0's regions before and after attention, in order. */
    private static List<Operation> regionLayer() {
        Qwen3PrefillRegions.Layer layer =
                Qwen3PrefillRegions.of(CONFIG, TOKENS, ROWS, Qwen3LayerSemanticsTest::type).get(0);
        List<Operation> ops = new ArrayList<>(layer.beforeAttention().ops());
        ops.addAll(layer.afterAttention().ops());
        return ops;
    }

    private static Projection projection(Region region) {
        MatMul m = (MatMul) region.ops().get(0);
        assertEquals(TOKENS, m.batchSize());
        return new Projection(m.weight().role(), m.rows(), m.columns());
    }
}
