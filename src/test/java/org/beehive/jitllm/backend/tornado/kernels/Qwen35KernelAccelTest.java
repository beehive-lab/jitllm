package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The {@code qwen35} mixer kernels, <b>compiled and executed on a device</b>, against the host
 * operations.
 *
 * <p>The parity tests beside this one run every lane on the host. That settles the arithmetic and
 * the addressing and says nothing about whether TornadoVM can compile the kernel — a distinction
 * that is not academic here. Q5_K's matrix-vector kernel passed its host parity test and then
 * failed to compile in seven successive formulations, the cause being one method call inlined into
 * a loop. These kernels use nested loops, a retained state array written in place, and {@code
 * TornadoMath} transcendentals, none of which the host tests exercise as device code.
 *
 * <p>Dimensions are the 27B's own where it matters — a 128-wide delta-net head, 48 value heads
 * against 16 key heads — and small in the ways that do not, so the test stays quick.
 */
public class Qwen35KernelAccelTest {

    private static final int STATE_DIM = 128;
    private static final int VALUE_HEADS = 48;
    private static final int KEY_HEADS = 16;
    private static final int KEY_DIM = KEY_HEADS * STATE_DIM;
    private static final int VALUE_DIM = VALUE_HEADS * STATE_DIM;
    private static final int CONV_DIM = 10240;
    private static final int CONV_KERNEL = 4;
    private static final float EPS = 1e-6f;

    private final Random random = new Random(20260908L);

    private float[] noise(int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) random.nextGaussian() * scale;
        }
        return v;
    }

    private static FloatArray toDevice(float[] v) {
        FloatArray a = new FloatArray(v.length);
        for (int i = 0; i < v.length; i++) {
            a.set(i, v[i]);
        }
        return a;
    }

    private static void run(TaskGraph graph, String taskName, int global, int local)
            throws Exception {
        WorkerGrid worker = new WorkerGrid1D(global);
        worker.setLocalWork(local, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(graph.getTaskGraphName() + "." + taskName, worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
    }

    private static void assertClose(String what, FloatTensor host, FloatArray device) {
        for (int i = 0; i < host.size(); i++) {
            float expected = host.getFloat(i);
            assertEquals(
                    what + "[" + i + "]",
                    expected,
                    device.get(i),
                    Math.max(1e-4f, Math.abs(expected) * 1e-4f));
        }
    }

    @Test
    public void theCausalConvolutionRunsOnTheDevice() throws Exception {
        float[] input = noise(CONV_DIM, 1.0f);
        float[] weight = noise(CONV_DIM * CONV_KERNEL, 0.5f);
        float[] window = noise(CONV_DIM * (CONV_KERNEL - 1), 1.0f);

        FloatTensor hostWindow = new ArrayFloatTensor(window.clone());
        FloatTensor hostOut = ArrayFloatTensor.allocate(CONV_DIM);
        CpuOperations.causalConv1d(
                new ArrayFloatTensor(input.clone()),
                new ArrayFloatTensor(weight.clone()),
                hostWindow,
                hostOut,
                CONV_DIM,
                CONV_KERNEL);

        FloatArray deviceWindow = toDevice(window);
        FloatArray deviceOut = new FloatArray(CONV_DIM);
        KernelContext context = new KernelContext();
        FloatArray deviceInput = toDevice(input);
        FloatArray deviceWeight = toDevice(weight);
        TaskGraph graph =
                new TaskGraph("conv")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                deviceInput,
                                deviceWeight,
                                deviceWindow,
                                deviceOut)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::causalConv1d,
                                context,
                                deviceInput,
                                deviceWeight,
                                deviceWindow,
                                deviceOut,
                                CONV_DIM,
                                CONV_KERNEL,
                                0)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, deviceOut, deviceWindow);
        run(graph, "k", CONV_DIM, 128);

        assertClose("conv out", hostOut, deviceOut);
        assertClose("conv window", hostWindow, deviceWindow);
    }

    @Test
    public void theDeltaRuleRunsOnTheDevice() throws Exception {
        float[] q = noise(KEY_DIM, 0.1f);
        float[] k = noise(KEY_DIM, 0.1f);
        float[] v = noise(VALUE_DIM, 1.0f);
        float[] state = noise(VALUE_HEADS * STATE_DIM * STATE_DIM, 0.05f);
        float[] decay = new float[VALUE_HEADS];
        float[] beta = new float[VALUE_HEADS];
        for (int h = 0; h < VALUE_HEADS; h++) {
            decay[h] = 0.5f + 0.5f * random.nextFloat();
            beta[h] = random.nextFloat();
        }

        FloatTensor hostState = new ArrayFloatTensor(state.clone());
        FloatTensor hostOut = ArrayFloatTensor.allocate(VALUE_DIM);
        CpuOperations.deltaRuleUpdate(
                new ArrayFloatTensor(q.clone()),
                new ArrayFloatTensor(k.clone()),
                new ArrayFloatTensor(v.clone()),
                new ArrayFloatTensor(decay.clone()),
                new ArrayFloatTensor(beta.clone()),
                hostState,
                hostOut,
                VALUE_HEADS,
                KEY_HEADS,
                STATE_DIM);

        FloatArray dq = toDevice(q);
        FloatArray dk = toDevice(k);
        FloatArray dv = toDevice(v);
        FloatArray ddecay = toDevice(decay);
        FloatArray dbeta = toDevice(beta);
        FloatArray dstate = toDevice(state);
        FloatArray dout = new FloatArray(VALUE_DIM);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("delta")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                dq,
                                dk,
                                dv,
                                ddecay,
                                dbeta,
                                dstate,
                                dout)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::deltaRule,
                                context,
                                dq,
                                dk,
                                dv,
                                ddecay,
                                dbeta,
                                dstate,
                                dout,
                                VALUE_HEADS,
                                KEY_HEADS,
                                STATE_DIM,
                                0)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dout, dstate);
        run(graph, "k", VALUE_HEADS * STATE_DIM, 128);

        assertClose("delta readout", hostOut, dout);
        assertClose("delta state", hostState, dstate);
    }

    // @formatter:off
    /**
     * The split delta rule — two lanes a column — against the same CPU reference.
     *
     * <p>It is the dispatched kernel wherever the value head's width is even, which is this
     * family's case, so the one-lane form above no longer covers what decode runs. Each column's
     * two reductions become a sum of two half-length folds, so this is checked against the
     * reference rather than against the other kernel's bits.
     *
     * <p><b>Ownership is what the state comparison pins.</b> Every state element is written once a
     * sweep by the lane owning its row range; an element left unwritten, or written twice, moves
     * away from the reference and fails here. The inputs carry the cases the ordinary noise above
     * does not reach: a head whose value is exactly zero, a head with beta zero so the correction
     * vanishes and the state only decays, a head whose keys and queries alternate exactly plus and
     * minus one so its dot products cancel term by term, and a decay of exactly one, which the
     * model's {@code exp(a * softplus(.))} can reach by rounding.
     */
    // @formatter:on
    @Test
    public void theSplitDeltaRuleRunsOnTheDevice() throws Exception {
        float[] q = noise(KEY_DIM, 0.1f);
        float[] k = noise(KEY_DIM, 0.1f);
        float[] v = noise(VALUE_DIM, 1.0f);
        float[] state = noise(VALUE_HEADS * STATE_DIM * STATE_DIM, 0.05f);
        float[] decay = new float[VALUE_HEADS];
        float[] beta = new float[VALUE_HEADS];
        for (int h = 0; h < VALUE_HEADS; h++) {
            decay[h] = 0.5f + 0.5f * random.nextFloat();
            beta[h] = random.nextFloat();
        }
        // Head 0's keys and queries cancel term by term; head 1 has no value; head 2 does not
        // update at all; head 3 sits on the decay boundary.
        for (int i = 0; i < STATE_DIM; i++) {
            k[i] = (i % 2 == 0) ? 1.0f : -1.0f;
            q[i] = (i % 2 == 0) ? 1.0f : -1.0f;
            v[STATE_DIM + i] = 0.0f;
        }
        beta[2] = 0.0f;
        decay[3] = 1.0f;

        FloatTensor hostState = new ArrayFloatTensor(state.clone());
        FloatTensor hostOut = ArrayFloatTensor.allocate(VALUE_DIM);
        CpuOperations.deltaRuleUpdate(
                new ArrayFloatTensor(q.clone()),
                new ArrayFloatTensor(k.clone()),
                new ArrayFloatTensor(v.clone()),
                new ArrayFloatTensor(decay.clone()),
                new ArrayFloatTensor(beta.clone()),
                hostState,
                hostOut,
                VALUE_HEADS,
                KEY_HEADS,
                STATE_DIM);

        FloatArray dq = toDevice(q);
        FloatArray dk = toDevice(k);
        FloatArray dv = toDevice(v);
        FloatArray ddecay = toDevice(decay);
        FloatArray dbeta = toDevice(beta);
        FloatArray dstate = toDevice(state);
        FloatArray dout = new FloatArray(VALUE_DIM);
        TaskGraph graph =
                new TaskGraph("deltaSplit")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                dq,
                                dk,
                                dv,
                                ddecay,
                                dbeta,
                                dstate,
                                dout)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::deltaRuleSplit,
                                new KernelContext(),
                                dq,
                                dk,
                                dv,
                                ddecay,
                                dbeta,
                                dstate,
                                dout,
                                KEY_HEADS,
                                STATE_DIM,
                                0)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dout, dstate);
        run(graph, "k", VALUE_HEADS * 2 * STATE_DIM, 2 * STATE_DIM);

        assertClose("split delta readout", hostOut, dout);
        assertClose("split delta state", hostState, dstate);
        for (int i = 0; i < VALUE_DIM; i++) {
            assertTrue("readout " + i + " is not finite", Float.isFinite(dout.get(i)));
        }

        // The eight-part kernel on the same inputs: the same host bounds, and its FP64 distance
        // no worse than the two-part kernel's by more than a small factor.
        FloatArray dstate8 = toDevice(state);
        FloatArray dout8 = new FloatArray(VALUE_DIM);
        TaskGraph graph8 =
                new TaskGraph("deltaSplit8")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                dq,
                                dk,
                                dv,
                                ddecay,
                                dbeta,
                                dstate8,
                                dout8)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::deltaRuleSplit8,
                                new KernelContext(),
                                dq,
                                dk,
                                dv,
                                ddecay,
                                dbeta,
                                dstate8,
                                dout8,
                                KEY_HEADS,
                                STATE_DIM,
                                0)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dout8, dstate8);
        run(
                graph8,
                "k",
                VALUE_HEADS * Qwen35DeltaNetKernels.DELTA_RULE_PARTS * STATE_DIM,
                Qwen35DeltaNetKernels.DELTA_RULE_PARTS * STATE_DIM);
        assertClose("eight-part delta readout", hostOut, dout8);
        assertClose("eight-part delta state", hostState, dstate8);
        double[] ref = deltaRuleFp64(q, k, v, decay, beta, state);
        double e2 = 0, e8 = 0, den = 0;
        for (int i = 0; i < VALUE_DIM; i++) {
            e2 += (dout.get(i) - ref[i]) * (dout.get(i) - ref[i]);
            e8 += (dout8.get(i) - ref[i]) * (dout8.get(i) - ref[i]);
            den += ref[i] * ref[i];
        }
        System.out.printf(
                java.util.Locale.ROOT,
                "[delta] readout vs FP64: two-part relL2 %.3e, eight-part relL2 %.3e%n",
                Math.sqrt(e2 / den),
                Math.sqrt(e8 / den));
        assertTrue("eight-part further from FP64 than two-part", e8 <= Math.max(4 * e2, 1e-20));
    }

    /** The delta rule's readout in FP64 from the same FP32 inputs. */
    private static double[] deltaRuleFp64(
            float[] q, float[] k, float[] v, float[] decay, float[] beta, float[] state) {
        double[] out = new double[VALUE_DIM];
        for (int h = 0; h < VALUE_HEADS; h++) {
            int kvBase = (h % KEY_HEADS) * STATE_DIM;
            double[][] s = new double[STATE_DIM][STATE_DIM];
            for (int i = 0; i < STATE_DIM; i++) {
                for (int c = 0; c < STATE_DIM; c++) {
                    s[i][c] =
                            (double) state[h * STATE_DIM * STATE_DIM + i * STATE_DIM + c]
                                    * decay[h];
                }
            }
            for (int c = 0; c < STATE_DIM; c++) {
                double pred = 0;
                for (int i = 0; i < STATE_DIM; i++) {
                    pred += s[i][c] * k[kvBase + i];
                }
                double corr = (v[h * STATE_DIM + c] - pred) * beta[h];
                double read = 0;
                for (int i = 0; i < STATE_DIM; i++) {
                    s[i][c] += k[kvBase + i] * corr;
                    read += s[i][c] * q[kvBase + i];
                }
                out[h * STATE_DIM + c] = read;
            }
        }
        return out;
    }

    @Test
    public void theGatedNormAndL2NormRunOnTheDevice() throws Exception {
        float[] values = noise(VALUE_DIM, 1.0f);
        float[] gate = noise(VALUE_DIM, 1.0f);
        float[] weight = noise(STATE_DIM, 1.0f);

        FloatTensor hostGated = new ArrayFloatTensor(values.clone());
        CpuOperations.gatedNorm(
                hostGated,
                new ArrayFloatTensor(gate.clone()),
                new ArrayFloatTensor(weight.clone()),
                VALUE_HEADS,
                STATE_DIM,
                EPS);

        FloatArray dvalues = toDevice(values);
        FloatArray dgate = toDevice(gate);
        FloatArray dweight = toDevice(weight);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("gated")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, dvalues, dgate, dweight)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::gatedNormPerHead,
                                context,
                                dvalues,
                                dgate,
                                dweight,
                                VALUE_HEADS,
                                STATE_DIM,
                                EPS)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dvalues);
        run(graph, "k", VALUE_HEADS, 16);
        assertClose("gated norm", hostGated, dvalues);

        float[] keys = noise(KEY_DIM, 1.0f);
        FloatTensor hostL2 = new ArrayFloatTensor(keys.clone());
        for (int head = 0; head < KEY_HEADS; head++) {
            CpuOperations.l2Norm(hostL2, head * STATE_DIM, STATE_DIM, EPS);
        }
        FloatArray dkeys = toDevice(keys);
        KernelContext l2Context = new KernelContext();
        TaskGraph l2 =
                new TaskGraph("l2")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, dkeys)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::l2NormPerHead,
                                l2Context,
                                dkeys,
                                KEY_HEADS,
                                STATE_DIM,
                                EPS)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dkeys);
        run(l2, "k", KEY_HEADS, 16);
        assertClose("l2", hostL2, dkeys);
    }

    // @formatter:off
    /**
     * The wide L2 norm — a workgroup per key head, a lane per element — against the same CPU
     * reference the per-head lane is held to, at the production geometry of 16 heads of 128.
     *
     * <p>It is the dispatched kernel wherever the key head's width is a power of two, which is this
     * family's case, so the per-head lane no longer covers what decode runs. The sum of squares is
     * a shared tree here rather than a left fold, so this is checked against the reference and not
     * against the other kernel's bits.
     *
     * <p>The epsilon clamps the norm — {@code 1 / max(sqrt(ss), eps)} — rather than sitting under
     * the root, which is not the gated norm's convention, so the inputs cover what that distinction
     * makes reachable: a head of exact zeros, a head far below the clamp, a head sitting on it, a
     * head whose elements cancel in sum but not in square, a single nonzero element, and a head
     * spanning twelve decades.
     */
    // @formatter:on
    @Test
    public void theWideL2NormRunsOnTheDevice() throws Exception {
        float[] keys = noise(KEY_DIM, 1.0f);
        for (int i = 0; i < STATE_DIM; i++) {
            keys[i] = 0.0f; // head 0: exact zeros
            keys[STATE_DIM + i] = (i % 2 == 0 ? 1 : -1) * 1.0e-24f; // head 1: below the clamp
            keys[2 * STATE_DIM + i] = 8.8e-8f; // head 2: on the clamp
            keys[3 * STATE_DIM + i] = (i % 2 == 0) ? 1.0f : -1.0f; // head 3: cancellation
            keys[4 * STATE_DIM + i] = i == 77 ? 3.0e4f : 1.0e-7f; // head 4: one large
            keys[5 * STATE_DIM + i] =
                    (float) ((i % 2 == 0 ? 1 : -1) * Math.pow(10.0, (i % 13) - 6));
            keys[6 * STATE_DIM + i] = i == 0 ? 2.5f : 0.0f; // head 6: one nonzero
        }

        FloatTensor host = new ArrayFloatTensor(keys.clone());
        for (int head = 0; head < KEY_HEADS; head++) {
            CpuOperations.l2Norm(host, head * STATE_DIM, STATE_DIM, EPS);
        }

        FloatArray dkeys = toDevice(keys);
        TaskGraph graph =
                new TaskGraph("l2Wide")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, dkeys)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::l2NormPerHeadWide,
                                new KernelContext(),
                                dkeys,
                                STATE_DIM,
                                EPS)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dkeys);
        run(graph, "k", KEY_DIM, STATE_DIM);
        assertClose("wide l2 norm", host, dkeys);
        for (int i = 0; i < STATE_DIM; i++) {
            assertEquals("the zero head must stay zero", 0.0f, dkeys.get(i), 0.0f);
        }
        for (int i = 0; i < KEY_DIM; i++) {
            assertTrue("element " + i + " is not finite", Float.isFinite(dkeys.get(i)));
        }
    }

    // @formatter:off
    /**
     * The wide gated norm — a workgroup per head, a lane per element — against the same CPU
     * reference the per-head lane is held to, at the production geometry of 48 heads of 128.
     *
     * <p>It is the dispatched kernel wherever the value head's width is a power of two, which is
     * this family's case, so the per-head lane above no longer covers what decode runs. The sum of
     * squares is a shared tree here rather than a left fold, so this is deliberately checked
     * against the reference and not against the other kernel's bits.
     *
     * <p>The inputs carry what the mapping could get wrong and the ordinary noise above would not
     * reach: a head of exact zeros, so the reduction bottoms out at the epsilon; a head whose
     * values are near the bottom of the float range; a head with one large element among tiny ones;
     * and gate values saturating the logistic at both ends, plus an exact zero.
     */
    // @formatter:on
    @Test
    public void theWideGatedNormRunsOnTheDevice() throws Exception {
        float[] values = noise(VALUE_DIM, 1.0f);
        float[] gate = noise(VALUE_DIM, 1.0f);
        float[] weight = noise(STATE_DIM, 1.0f);
        for (int i = 0; i < STATE_DIM; i++) {
            values[i] = 0.0f; // head 0: all zero
            values[STATE_DIM + i] = (i % 2 == 0 ? 1 : -1) * 1.0e-21f; // head 1: near zero
            values[2 * STATE_DIM + i] = i == 63 ? 3.0e4f : 1.0e-7f; // head 2: one large
        }
        for (int i = 0; i < VALUE_DIM; i++) {
            int mode = i % 5;
            if (mode == 0) {
                gate[i] = 90.0f;
            } else if (mode == 1) {
                gate[i] = -90.0f;
            } else if (mode == 2) {
                gate[i] = 0.0f;
            }
        }

        FloatTensor host = new ArrayFloatTensor(values.clone());
        CpuOperations.gatedNorm(
                host,
                new ArrayFloatTensor(gate.clone()),
                new ArrayFloatTensor(weight.clone()),
                VALUE_HEADS,
                STATE_DIM,
                EPS);

        FloatArray dvalues = toDevice(values);
        FloatArray dgate = toDevice(gate);
        FloatArray dweight = toDevice(weight);
        TaskGraph graph =
                new TaskGraph("gatedWide")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, dvalues, dgate, dweight)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::gatedNormPerHeadWide,
                                new KernelContext(),
                                dvalues,
                                dgate,
                                dweight,
                                STATE_DIM,
                                EPS)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dvalues);
        run(graph, "k", VALUE_DIM, STATE_DIM);
        assertClose("wide gated norm", host, dvalues);
        for (int i = 0; i < VALUE_DIM; i++) {
            assertTrue("element " + i + " is not finite", Float.isFinite(dvalues.get(i)));
        }
    }

    @Test
    public void theDecayAndBetaRunOnTheDevice() throws Exception {
        float[] alpha = noise(VALUE_HEADS, 1.0f);
        float[] beta = noise(VALUE_HEADS, 1.0f);
        float[] dtBias = noise(VALUE_HEADS, 0.5f);
        float[] a = new float[VALUE_HEADS];
        for (int h = 0; h < VALUE_HEADS; h++) {
            a[h] = -(0.5f + random.nextFloat());
        }

        float[] expectedDecay = new float[VALUE_HEADS];
        float[] expectedBeta = new float[VALUE_HEADS];
        for (int h = 0; h < VALUE_HEADS; h++) {
            expectedBeta[h] = CpuOperations.logistic(beta[h]);
            expectedDecay[h] =
                    (float) Math.exp(a[h] * CpuOperations.softplus(alpha[h] + dtBias[h]));
        }

        FloatArray dalpha = toDevice(alpha);
        FloatArray dbeta = toDevice(beta);
        FloatArray ddt = toDevice(dtBias);
        FloatArray da = toDevice(a);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("decay")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, dalpha, dbeta, ddt, da)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::decayAndBeta,
                                context,
                                dalpha,
                                dbeta,
                                ddt,
                                da,
                                VALUE_HEADS)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dalpha, dbeta);
        run(graph, "k", VALUE_HEADS, 16);

        for (int h = 0; h < VALUE_HEADS; h++) {
            assertEquals("beta[" + h + "]", expectedBeta[h], dbeta.get(h), 1e-5f);
            assertEquals("decay[" + h + "]", expectedDecay[h], dalpha.get(h), 1e-5f);
        }
    }

    @Test
    public void theAttentionKernelsRunOnTheDevice() throws Exception {
        int heads = 24;
        int kvHeads = 4;
        int headDim = 256;
        int rotaryDim = 64;
        int queryDim = heads * headDim;
        int kvDim = kvHeads * headDim;
        int position = 5;

        float[] fused = noise(queryDim * 2, 1.0f);
        FloatArray dfused = toDevice(fused);
        FloatArray dquery = new FloatArray(queryDim);
        FloatArray dgate = new FloatArray(queryDim);
        KernelContext splitContext = new KernelContext();
        TaskGraph split =
                new TaskGraph("split")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, dfused, dquery, dgate)
                        .task(
                                "k",
                                Qwen35AttentionKernels::splitQueryGate,
                                splitContext,
                                dfused,
                                dquery,
                                dgate,
                                heads,
                                headDim)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dquery, dgate);
        run(split, "k", queryDim, 128);
        for (int head = 0; head < heads; head++) {
            for (int i = 0; i < headDim; i += 37) {
                assertEquals(
                        "query", fused[head * 2 * headDim + i], dquery.get(head * headDim + i), 0f);
                assertEquals(
                        "gate",
                        fused[head * 2 * headDim + headDim + i],
                        dgate.get(head * headDim + i),
                        0f);
            }
        }

        var freqs =
                org.beehive.jitllm.model.loader.RopeFrequencies.precomputeFreqsCis(
                        64, rotaryDim, 1e7f, false, 0, 0, 0, 0);
        float[] query = noise(queryDim, 1.0f);
        float[] key = noise(kvDim, 1.0f);
        FloatTensor hostQuery = new ArrayFloatTensor(query.clone());
        FloatTensor hostKey = new ArrayFloatTensor(key.clone());
        CpuOperations.ropeNeoxPartial(
                hostQuery,
                heads,
                headDim,
                rotaryDim,
                position,
                new ArrayFloatTensor(freqs.first()),
                new ArrayFloatTensor(freqs.second()));
        CpuOperations.ropeNeoxPartial(
                hostKey,
                kvHeads,
                headDim,
                rotaryDim,
                position,
                new ArrayFloatTensor(freqs.first()),
                new ArrayFloatTensor(freqs.second()));

        FloatArray dq = toDevice(query);
        FloatArray dk = toDevice(key);
        IntArray positionHolder = new IntArray(2);
        positionHolder.set(0, position);
        positionHolder.set(1, 0);
        KernelContext ropeContext = new KernelContext();
        FloatArray dreal = toDevice(freqs.first());
        FloatArray dimag = toDevice(freqs.second());
        TaskGraph rope =
                new TaskGraph("rope")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                positionHolder,
                                dq,
                                dk,
                                dreal,
                                dimag)
                        .task(
                                "k",
                                Qwen35AttentionKernels::ropeNeoxPartial,
                                ropeContext,
                                positionHolder,
                                dq,
                                dk,
                                dreal,
                                dimag,
                                heads,
                                kvHeads,
                                headDim,
                                rotaryDim)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dq, dk);
        run(rope, "k", heads * (rotaryDim / 2), 32);
        assertClose("rope query", hostQuery, dq);
        assertClose("rope key", hostKey, dk);

        // The output gate.
        float[] values = noise(queryDim, 1.0f);
        float[] gateValues = noise(queryDim, 1.0f);
        FloatArray dvalues = toDevice(values);
        FloatArray dgateValues = toDevice(gateValues);
        KernelContext gateContext = new KernelContext();
        TaskGraph gate =
                new TaskGraph("gate")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, dvalues, dgateValues)
                        .task(
                                "k",
                                Qwen35AttentionKernels::applyOutputGate,
                                gateContext,
                                dvalues,
                                dgateValues,
                                queryDim)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dvalues);
        run(gate, "k", queryDim, 128);
        for (int i = 0; i < queryDim; i += 53) {
            float expected = values[i] * CpuOperations.logistic(gateValues[i]);
            assertTrue(
                    "gated[" + i + "] " + expected + " vs " + dvalues.get(i),
                    Math.abs(expected - dvalues.get(i))
                            <= Math.max(1e-5f, Math.abs(expected) * 1e-5f));
        }
    }
}
