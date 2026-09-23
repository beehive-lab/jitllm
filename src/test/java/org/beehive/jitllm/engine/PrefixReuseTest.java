package org.beehive.jitllm.engine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.runtime.batch.BatchSlots;
import org.beehive.jitllm.runtime.kv.BlockPool;
import org.beehive.jitllm.runtime.kv.KvCacheManager;
import org.junit.Test;

/**
 * The claim a prefix cache makes is that the device never runs the positions a previous sequence
 * already ran. That is countable: a recording executor sees every position it was asked to decode,
 * so a second request sharing a prefix should simply not ask for the covered ones.
 */
public class PrefixReuseTest {

    private static final int BLOCK_TOKENS = 4;

    /** Records the positions it was asked to decode, per slot. */
    private static final class RecordingExecutor
            implements org.beehive.jitllm.runtime.batch.BatchExecutor {

        final List<Integer> decodedPositions = new ArrayList<>();
        private final int batchSize;

        RecordingExecutor(int batchSize) {
            this.batchSize = batchSize;
        }

        @Override
        public int maxBatchSize() {
            return batchSize;
        }

        @Override
        public int[] decodeStep(BatchSlots batch) {
            int[] tokens = new int[batch.width()];
            for (int i = 0; i < batch.width(); i++) {
                if (batch.active()[i]) {
                    decodedPositions.add(batch.positions()[i]);
                    tokens[i] = 500 + batch.positions()[i];
                }
            }
            return tokens;
        }

        @Override
        public boolean isStopToken(int token) {
            return false;
        }
    }

    @Test
    public void aSharedPrefixIsNeverDecodedTwice() {
        BlockPool pool = new BlockPool(4 * 4, 4, 4, BLOCK_TOKENS, 1024);
        try (KvCacheManager manager = new KvCacheManager(pool)) {
            manager.enablePrefixCache(4);
            RecordingExecutor executor = new RecordingExecutor(1);

            // Eight prompt tokens: two whole blocks, so the whole prompt is shareable.
            int[] prompt = {11, 12, 13, 14, 15, 16, 17, 18};

            try (LLMEngine engine =
                    new LLMEngine(TestModels.sharedKvCapable(), manager, executor, 1, 8)) {
                RequestHandle first = engine.addRequest(prompt, 2, null);
                while (!first.isTerminal()) {
                    engine.step();
                }
                int decodedByFirst = executor.decodedPositions.size();
                executor.decodedPositions.clear();

                RequestHandle second = engine.addRequest(prompt, 2, null);
                while (!second.isTerminal()) {
                    engine.step();
                }
                int decodedBySecond = executor.decodedPositions.size();

                assertTrue(
                        "the second request ran fewer steps than the first",
                        decodedBySecond < decodedByFirst);
                // The prefix covers the whole prompt, but the last prompt token is always re-fed:
                // the cache holds its KV, not the logits the first generated token comes from.
                assertEquals(
                        "it skipped every cached position but the last",
                        decodedByFirst - (prompt.length - 1),
                        decodedBySecond);
                assertEquals(
                        "and it started at the last prompt position, not at zero",
                        prompt.length - 1,
                        (int) executor.decodedPositions.get(0));
                assertEquals(
                        "both produced the tokens they were asked for",
                        first.tokenCount(),
                        second.tokenCount());

                assertEquals(1, manager.prefixCache().hits());
                assertEquals(
                        "two blocks of prefill not repeated",
                        2,
                        manager.prefixCache().blocksReused());
            }
        }
    }

    /** With no prefix cache the engine behaves exactly as before — nothing is skipped. */
    @Test
    public void withoutAPrefixCacheNothingIsSkipped() {
        BlockPool pool = new BlockPool(4 * 4, 4, 4, BLOCK_TOKENS, 1024);
        try (KvCacheManager manager = new KvCacheManager(pool)) {
            RecordingExecutor executor = new RecordingExecutor(1);
            int[] prompt = {11, 12, 13, 14, 15, 16, 17, 18};

            try (LLMEngine engine =
                    new LLMEngine(TestModels.sharedKvCapable(), manager, executor, 1, 8)) {
                RequestHandle first = engine.addRequest(prompt, 2, null);
                while (!first.isTerminal()) {
                    engine.step();
                }
                int decodedByFirst = executor.decodedPositions.size();
                executor.decodedPositions.clear();

                RequestHandle second = engine.addRequest(prompt, 2, null);
                while (!second.isTerminal()) {
                    engine.step();
                }

                assertEquals(
                        "the same work, twice", decodedByFirst, executor.decodedPositions.size());
                assertEquals(
                        "starting from the beginning", 0, (int) executor.decodedPositions.get(0));
            }
        }
    }
}
