package org.beehive.jllm.backend.tornado.lowering;

/**
 * A lowering's refusal, naming what it expected and what it was handed.
 *
 * <p>Top-level rather than nested in one family's lowering, because there are two families now and
 * a refusal is the same event whichever raised it. What differs is the {@code family} prefix, which
 * is why it is a parameter rather than a constant in the message.
 *
 * <p>The message names <b>both sides</b> on purpose. "unsupported program" tells whoever reads the
 * log nothing they can act on; "layer_0 component 4 rotary layout expected NEOX but found
 * INTERLEAVED" tells them which description to look at and which line of it is wrong.
 */
public final class UnsupportedProgramException extends RuntimeException {

    UnsupportedProgramException(String family, String what, String expected, String found) {
        super(
                "this lowering handles "
                        + family
                        + " only; "
                        + what
                        + " expected "
                        + expected
                        + " but found "
                        + found);
    }
}
