package org.beehive.jitllm.auxiliary;

/** mask of a byte-sequence in UTF-8 encoding */
public record Utf8Mask(int mask, int pattern, int len) {
    // @formatter:off
    public static final Utf8Mask[] MASKS = {
        new Utf8Mask(0b11100000, 0b11000000, 2),
        new Utf8Mask(0b11110000, 0b11100000, 3),
        new Utf8Mask(0b11111000, 0b11110000, 4)
    };
    // @formatter:on
}
