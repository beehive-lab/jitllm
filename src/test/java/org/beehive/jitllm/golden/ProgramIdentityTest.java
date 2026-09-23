package org.beehive.jllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;

/**
 * {@link CompiledProgramIdentityAccelTest} can only report "nothing changed", so on its own it
 * would pass just as happily if it observed nothing at all. These tests pin the parts that decide
 * what it observes: that kernel source is recognised and hashed, that a changed kernel produces a
 * different snapshot, and that the recorder counts one capture per compiled task while leaving
 * ordinary output alone.
 */
public class ProgramIdentityTest {

    private static final String CUDA_KERNEL =
            """
            extern "C" __global__ void reductionOneBlockWithLayer(long long *_kernel_context,
                unsigned char *output, unsigned char *x, int size) {
              int gid = blockIdx.x * blockDim.x + threadIdx.x;
              // padding so the source is long enough to be recognised as a kernel
              if (gid < size) { output[gid] = x[gid]; }
            }
            """;

    private static final String OPENCL_KERNEL =
            """
            __kernel void ropeRotationWithCacheCopyPrecomputed(__global long *_kernel_context,
                __global uchar *sq, __global uchar *sk, int kvDim) {
              int gid = get_global_id(0);
              // padding so the source is long enough to be recognised as a kernel
              if (gid < kvDim) { sq[gid] = sk[gid]; }
            }
            """;

    private static final String PTX_KERNEL =
            """
            //
            .visible .entry fusedQKVMatmulQ8(
              .param .u64 param_0, .param .u64 param_1
            ) {
              // padding so the source is long enough to be recognised as a kernel
              ret;
            }
            """;

    @Test
    public void entryPointsAreParsedForEveryBackendSourceLanguage() {
        assertEquals(
                List.of("reductionOneBlockWithLayer"), ProgramIdentity.entryPoints(CUDA_KERNEL));
        assertEquals(
                List.of("ropeRotationWithCacheCopyPrecomputed"),
                ProgramIdentity.entryPoints(OPENCL_KERNEL));
        assertEquals(List.of("fusedQKVMatmulQ8"), ProgramIdentity.entryPoints(PTX_KERNEL));
    }

    @Test
    public void aModuleWithSeveralEntryPointsKeepsDeclarationOrder() {
        String twoKernels =
                CUDA_KERNEL
                        + "\n"
                        + CUDA_KERNEL.replace("reductionOneBlockWithLayer", "rmsNormReduce");
        assertEquals(
                List.of("reductionOneBlockWithLayer", "rmsNormReduce"),
                ProgramIdentity.entryPoints(twoKernels));
    }

    /** Source with no recognisable entry point still hashes, under a stable placeholder name. */
    @Test
    public void unnamedSourceGetsAStablePlaceholderName() {
        List<String> names = ProgramIdentity.entryPoints("some source with no entry point at all");
        assertEquals(1, names.size());
        assertTrue(names.get(0), names.get(0).startsWith("<unnamed:"));
        assertEquals(names, ProgramIdentity.entryPoints("some source with no entry point at all"));
    }

    @Test
    public void aChangedKernelChangesTheSnapshot() {
        GridScheduler scheduler = scheduler();
        ProgramIdentity.Snapshot before =
                ProgramIdentity.snapshot(2, List.of(CUDA_KERNEL, OPENCL_KERNEL), scheduler);

        // One instruction changed inside one kernel: same task names, different hash.
        String patched = CUDA_KERNEL.replace("output[gid] = x[gid];", "output[gid] = x[gid] + 1;");
        ProgramIdentity.Snapshot after =
                ProgramIdentity.snapshot(2, List.of(patched, OPENCL_KERNEL), scheduler);

        assertEquals(before.taskNames(), after.taskNames());
        assertNotEquals(
                "a changed kernel must change its source hash",
                before.sourceHashes(),
                after.sourceHashes());
    }

    @Test
    public void aRenamedKernelChangesTheTaskNames() {
        GridScheduler scheduler = scheduler();
        ProgramIdentity.Snapshot before =
                ProgramIdentity.snapshot(1, List.of(CUDA_KERNEL), scheduler);
        ProgramIdentity.Snapshot after =
                ProgramIdentity.snapshot(
                        1,
                        List.of(
                                CUDA_KERNEL.replace(
                                        "reductionOneBlockWithLayer", "reductionSingleGroup")),
                        scheduler);
        assertNotEquals(before.taskNames(), after.taskNames());
    }

    @Test
    public void gridEntriesAreSortedAndCarryTheirWorkDimensions() {
        GridScheduler scheduler = new GridScheduler();
        WorkerGrid second = new WorkerGrid1D(256);
        second.setLocalWork(32, 1, 1);
        scheduler.addWorkerGrid("graph.zeta", second);
        WorkerGrid first = new WorkerGrid1D(1024);
        first.setLocalWork(128, 1, 1);
        scheduler.addWorkerGrid("graph.alpha", first);

        List<String> entries = ProgramIdentity.gridEntries(scheduler);
        assertEquals(2, entries.size());
        assertTrue(entries.get(0), entries.get(0).startsWith("graph.alpha "));
        assertTrue(entries.get(0), entries.get(0).contains("global=[1024, 1, 1]"));
        assertTrue(entries.get(0), entries.get(0).contains("local=[128, 1, 1]"));
        assertTrue(entries.get(1), entries.get(1).startsWith("graph.zeta "));
    }

    /**
     * A task relaunched with a different grid is a different program, and must not compare equal.
     */
    @Test
    public void aChangedWorkerGridChangesTheEntries() {
        GridScheduler before = new GridScheduler("graph.task", new WorkerGrid1D(1024));
        GridScheduler after = new GridScheduler("graph.task", new WorkerGrid1D(512));
        assertNotEquals(ProgramIdentity.gridEntries(before), ProgramIdentity.gridEntries(after));
    }

    @Test
    public void theRecorderKeepsKernelSourceAndPassesEverythingElseThrough() {
        ByteArrayOutputStream passedThrough = new ByteArrayOutputStream();
        ProgramIdentity.SourceRecorder recorder =
                new ProgramIdentity.SourceRecorder(
                        new PrintStream(passedThrough, true, StandardCharsets.UTF_8));

        recorder.println("[INFO] ordinary output");
        recorder.println(CUDA_KERNEL);
        recorder.println(OPENCL_KERNEL);
        recorder.println("[INFO] more ordinary output");

        assertEquals("one capture per compiled task", 2, recorder.count());
        assertEquals(List.of(CUDA_KERNEL, OPENCL_KERNEL), recorder.sources());

        String forwarded = passedThrough.toString(StandardCharsets.UTF_8);
        assertTrue(forwarded, forwarded.contains("[INFO] ordinary output"));
        assertTrue(forwarded, forwarded.contains("[INFO] more ordinary output"));
        assertTrue(
                "kernel source must not be echoed to the console",
                !forwarded.contains("__global__"));
    }

    /** A short line that merely mentions a kernel keyword is log output, not a compiled module. */
    @Test
    public void theRecorderDoesNotMistakeLogLinesForKernelSource() {
        ByteArrayOutputStream passedThrough = new ByteArrayOutputStream();
        ProgramIdentity.SourceRecorder recorder =
                new ProgramIdentity.SourceRecorder(
                        new PrintStream(passedThrough, true, StandardCharsets.UTF_8));
        recorder.println("installing __kernel void saxpy");
        assertEquals(0, recorder.count());
    }

    private static GridScheduler scheduler() {
        return new GridScheduler("graph.task", new WorkerGrid1D(1024));
    }
}
