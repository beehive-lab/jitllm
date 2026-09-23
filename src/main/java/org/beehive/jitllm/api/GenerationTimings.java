package org.beehive.jllm.api;

import java.time.Duration;

/**
 * How long a generation took, and how fast it ran.
 *
 * <p>Immutable and thread-safe. This is the public replacement for the CLI's printed metrics: the
 * same numbers, returned rather than printed, so a program can act on them.
 *
 * <p>Rates are derived from the counts and durations rather than stored, so they cannot disagree
 * with them. A phase that took no measurable time reports a rate of zero rather than infinity — a
 * run too short to time is not a run of infinite speed.
 */
@Experimental
public final class GenerationTimings {

    private static final double NANOS_PER_SECOND = 1e9;

    private final Duration prefill;
    private final Duration decode;
    private final int promptTokens;
    private final int generatedTokens;

    public GenerationTimings(
            Duration prefill, Duration decode, int promptTokens, int generatedTokens) {
        this.prefill = prefill == null ? Duration.ZERO : prefill;
        this.decode = decode == null ? Duration.ZERO : decode;
        this.promptTokens = promptTokens;
        this.generatedTokens = generatedTokens;
    }

    /** Time spent consuming the prompt. */
    public Duration prefill() {
        return prefill;
    }

    /** Time spent generating. */
    public Duration decode() {
        return decode;
    }

    /** Prompt tokens processed per second, or 0 when the phase was too short to time. */
    public double promptTokensPerSecond() {
        return rate(promptTokens, prefill);
    }

    /** Generated tokens per second — the throughput number that gets compared between runs. */
    public double generatedTokensPerSecond() {
        return rate(generatedTokens, decode);
    }

    private static double rate(int tokens, Duration duration) {
        long nanos = duration.toNanos();
        if (tokens <= 0 || nanos <= 0) {
            return 0.0;
        }
        return tokens / (nanos / NANOS_PER_SECOND);
    }

    @Override
    public String toString() {
        return String.format(
                "prefill %d tok in %s (%.2f tok/s), decode %d tok in %s (%.2f tok/s)",
                promptTokens,
                prefill,
                promptTokensPerSecond(),
                generatedTokens,
                decode,
                generatedTokensPerSecond());
    }
}
