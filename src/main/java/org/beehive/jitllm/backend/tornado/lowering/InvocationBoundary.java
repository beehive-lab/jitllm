package org.beehive.jitllm.backend.tornado.lowering;

import org.beehive.jitllm.inference.Logits;

/**
 * The boundary a lowered, workspace-sharing compiled program is invoked through.
 *
 * <p>Everything a session gives the device goes in as a <b>logical value</b>, and everything it
 * gets back is <b>session-owned</b>. Nothing in between escapes: the domain's workspace is not
 * reachable from {@code InferenceCore}, {@code TokenGenerationLoop}, a session or the public API,
 * and there is no accessor here that would make it so.
 *
 * <p>One call does the whole turn, with the compiled entry's lock held across it:
 *
 * <pre>
 *   lock
 *     → stage the inputs into the domain-owned carriers
 *     → execute
 *     → copy the required results out into session-owned storage
 *   unlock
 * </pre>
 */
public interface InvocationBoundary {

    /**
     * One invocation: stage, execute, copy out.
     *
     * @param token the token being processed — an invocation value, staged into the domain's
     *     embedding carrier rather than into any session-owned array
     * @param position its position in the sequence
     * @return this session's own results; never a view into domain-owned storage
     */
    Result invoke(int token, int position);

    /**
     * What one invocation produced, in storage the calling session owns.
     *
     * @param logits this session's copy of the logits, as the neutral view. The view is over
     *     session-owned storage, built once and reused, so it is no more a window onto the domain's
     *     carriers than the array was
     * @param sampledToken the token the device chose, or {@code -1} when sampling is host-resident
     */
    record Result(Logits logits, int sampledToken) {

        /** Whether the device already chose a token, making a host sample unnecessary. */
        public boolean hasSampledToken() {
            return sampledToken >= 0;
        }
    }
}
