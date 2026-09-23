package org.beehive.jllm.auxiliary;

import java.util.function.IntConsumer;
import java.util.function.LongConsumer;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

public final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }

    public static void parallelForLong(
            long startInclusive, long endExclusive, LongConsumer action) {
        if (startInclusive == 0 && endExclusive == 1) {
            action.accept(0);
            return;
        }
        LongStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }
}
