package org.beehive.jllm.golden;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * The provisional reproducibility-envelope gate ({@code verification-gates.md}
 * §Reproducibility-envelope gate).
 *
 * <p>Applies to a configuration recorded as {@code bit_exact: false}. It bounds how far the numbers
 * may drift and asserts the properties that actually change output. It is an accommodation of a
 * known defect, not a relaxed standard: argmax and top-5 must hold exactly, and top-10 is reported
 * so that a widening drift is visible rather than silent.
 */
public final class Envelope {

    // Bounds are empirical, set with headroom over the drift measured on the reference tuple
    // (max absolute drift 0.05.0.47 across repeated runs). They exist to catch the defect
    // getting worse, not to bless its current size.

    /** Absolute drift permitted per element. About 2x the worst observed. */
    public static final double MAX_ABS = 1.0;

    /**
     * Relative drift permitted, applied only above {@link #REL_FLOOR}. At the floor, the worst
     * observed absolute drift (0.47) is ~0.09 relative, so 0.15 leaves headroom without being
     * vacuous.
     */
    public static final double MAX_REL = 0.15;

    /**
     * Below this magnitude an element is dominated by absolute noise and the relative bound would
     * be redundant with {@link #MAX_ABS}. Elements at or above it are the ones that decide argmax
     * and top-k, which is where relative accuracy actually matters.
     */
    public static final double REL_FLOOR = 5.0;

    public static final class Report {
        public int changed;
        public double maxAbs;
        public double maxRel;
        public int argmaxRef;
        public int argmaxGot;
        public boolean top5Same;
        public boolean top10Same;
        public int total;

        public boolean withinBounds() {
            return maxAbs <= MAX_ABS && maxRel <= MAX_REL && argmaxRef == argmaxGot && top5Same;
        }

        @Override
        public String toString() {
            return String.format(
                    "changed=%d/%d maxAbs=%.6g (limit %.3g) maxRel=%.6g (limit %.3g) argmax %d->%d top5=%s top10=%s",
                    changed,
                    total,
                    maxAbs,
                    MAX_ABS,
                    maxRel,
                    MAX_REL,
                    argmaxRef,
                    argmaxGot,
                    top5Same ? "same" : "CHANGED",
                    top10Same ? "same" : "CHANGED");
        }
    }

    private Envelope() {}

    public static Report compare(float[] reference, float[] got) {
        Report r = new Report();
        r.total = reference.length;
        for (int i = 0; i < reference.length; i++) {
            double d = Math.abs((double) reference[i] - (double) got[i]);
            if (Float.floatToRawIntBits(reference[i]) != Float.floatToRawIntBits(got[i])) {
                r.changed++;
            }
            r.maxAbs = Math.max(r.maxAbs, d);
            double ref = Math.abs(reference[i]);
            if (ref >= REL_FLOOR) {
                r.maxRel = Math.max(r.maxRel, d / ref);
            }
        }
        r.argmaxRef = argmax(reference);
        r.argmaxGot = argmax(got);
        r.top5Same = topK(reference, 5).equals(topK(got, 5));
        r.top10Same = topK(reference, 10).equals(topK(got, 10));
        return r;
    }

    public static boolean hasNonFinite(float[] v) {
        for (float f : v) {
            if (Float.isNaN(f) || Float.isInfinite(f)) {
                return true;
            }
        }
        return false;
    }

    public static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    public static Set<Integer> topK(float[] v, int k) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (p, q) -> Float.compare(v[q], v[p]));
        Set<Integer> out = new HashSet<>();
        for (int i = 0; i < k && i < idx.length; i++) {
            out.add(idx[i]);
        }
        return out;
    }
}
