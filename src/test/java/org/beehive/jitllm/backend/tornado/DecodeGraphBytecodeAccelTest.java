package org.beehive.jllm.backend.tornado;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.beehive.jllm.Options;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.junit.Test;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

// @formatter:off
/**
 * How much of TornadoVM's fixed per-graph bytecode buffer the decode layer graphs actually use.
 *
 * <p>{@code TornadoTaskGraph} encodes each graph's high-level bytecode into a {@code private
 * byte[8192]} allocated in its constructor, and there is no growth path: a graph that needs more
 * throws {@code BufferOverflowException} while it is being built. That is a <b>hard capacity
 * limit</b>, and it is what bounds how many transformer layers can share one decode graph. It is
 * not the buffer {@code tornado.tvm.maxbytecodesize} sizes, which is a different one further down.
 *
 * <p>Layers per graph is a performance decision — each graph costs one submission and one device
 * wait, measured at about 47 µs — so the real headroom matters. This measures it rather than
 * assuming it, and fails if a change to the layer body eats the margin, which would otherwise
 * surface as an overflow on a model with more layers in a group.
 *
 * <p>Read by reflection because the buffer is private with no accessor; the alternative is to
 * discover the limit by building graphs until one throws.
 */
// @formatter:on
public class DecodeGraphBytecodeAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final String KV_FP16_PROPERTY = "jllm.kvcache.fp16";
    private static final int BATCH = 128;
    private static final int CONTEXT = 512;

    /** {@code TornadoTaskGraph.highLevelCode} is allocated at this size and never grows. */
    private static final int BUFFER_CAPACITY = 8192;

    /**
     * How much of the buffer any one graph may use before this fails. Not a tuning knob: the margin
     * is what absorbs a layer gaining a task, and losing it means the overflow lands on somebody
     * else's model rather than here.
     */
    private static final double MAX_FRACTION = 0.90;

    @Test
    public void theDecodeLayerGraphsFitTheirBytecodeBufferWithMargin() throws Exception {
        Path model = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                model != null);

        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousKv = System.getProperty(KV_FP16_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        System.setProperty(KV_FP16_PROPERTY, "true");
        try {
            Options options =
                    new Options(
                            model,
                            "bytecode",
                            null,
                            null,
                            false,
                            0.0f,
                            1.0f,
                            42,
                            CONTEXT,
                            false,
                            false,
                            true,
                            true,
                            BATCH);
            Model loaded = ModelLoader.loadModel(options);
            State state = loaded.createNewState();
            TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, loaded);
            try {
                Map<String, Integer> used = encodedSizes(plan);
                assertTrue("no task graph could be read from the plan", !used.isEmpty());

                int worst = 0;
                String worstName = "";
                for (Map.Entry<String, Integer> e : used.entrySet()) {
                    System.out.printf(
                            "[BYTECODE] %-30s %5d of %d bytes (%.0f%%)%n",
                            e.getKey(),
                            e.getValue(),
                            BUFFER_CAPACITY,
                            100.0 * e.getValue() / BUFFER_CAPACITY);
                    if (e.getValue() > worst) {
                        worst = e.getValue();
                        worstName = e.getKey();
                    }
                }
                assertTrue(
                        "the largest graph, "
                                + worstName
                                + ", encodes "
                                + worst
                                + " of "
                                + BUFFER_CAPACITY
                                + " bytes. TornadoTaskGraph's buffer does not grow, so the margin"
                                + " above this is all there is: either a graph must hold less or"
                                + " fewer layers must share one.",
                        worst <= MAX_FRACTION * BUFFER_CAPACITY);
            } finally {
                plan.freeTornadoExecutionPlan();
            }
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(KV_FP16_PROPERTY, previousKv);
        }
    }

    /** Encoded high-level bytecode size per task graph, read out of the private buffer. */
    static Map<String, Integer> encodedSizes(TornadoVMMasterPlan plan) throws Exception {
        Map<String, Integer> out = new LinkedHashMap<>();
        var forward = PlanDispatchEvidence.forwardPlanIfAvailable(plan);
        if (forward == null) {
            return out;
        }
        for (ImmutableTaskGraph immutable : forward) {
            Object inner = read(immutable, "taskGraph");
            Object impl = read(inner, "taskGraphImpl");
            ByteBuffer hl = (ByteBuffer) read(impl, "hlBuffer");
            String name = String.valueOf(read(impl, "taskGraphName"));
            out.put(name, hl.position());
        }
        return out;
    }

    private static Object read(Object target, String field) throws Exception {
        Class<?> c = target.getClass();
        while (c != null) {
            try {
                Field f = c.getDeclaredField(field);
                f.setAccessible(true);
                return f.get(target);
            } catch (NoSuchFieldException ignored) {
                c = c.getSuperclass();
            }
        }
        throw new NoSuchFieldException(field + " on " + target.getClass());
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
