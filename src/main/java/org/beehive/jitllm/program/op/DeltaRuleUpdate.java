package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * One step of the gated delta rule, for every value head.
 *
 * <p>The recurrent counterpart of {@link Attention}. Where attention scores a query against a
 * growing key/value store, this keeps one {@code stateDim × stateDim} matrix per head into which
 * the whole history has been summed, and per step:
 *
 * <pre>
 *   S      *= decay[h]                    // forget
 *   d       = (v[h] - Sᵀ·k[h]) * beta[h]  // what the state does not already predict
 *   S      += k[h] ⊗ d                    // write the correction
 *   out[h]  = Sᵀ·q[h]                     // read it back
 * </pre>
 *
 * <p>Correcting only towards what the state does not already produce is the delta rule proper, and
 * is what keeps a fixed-size state from saturating; {@code beta} is how much of that correction to
 * apply. It is one operation and not a composition of matrix multiplies and adds because the update
 * and the readout share the state and are ordered with respect to it — a decomposition would name
 * neither, and would leave a backend nothing to fuse.
 *
 * <p><b>Key heads may be fewer than value heads, and value head {@code h} reads key head {@code h %
 * keyHeads}</b> — the reference repeats the key heads by tiling, not by blocking. Stating {@code
 * keyHeads} rather than a ratio is deliberate: a ratio invites the division that produces the wrong
 * pairing.
 *
 * @param query queries, {@code keyHeads * stateDim}, already normalized and scaled
 * @param key keys, {@code keyHeads * stateDim}, already normalized
 * @param value values, {@code valueHeads * stateDim}
 * @param decay per value head, already exponentiated
 * @param beta per value head, already through the logistic
 * @param state the retained matrices, updated in place
 * @param output the readout, {@code valueHeads * stateDim}
 * @param valueHeads how many value heads
 * @param keyHeads how many key heads there are to cycle through
 * @param stateDim the head width, equal for keys and values
 * @param dataType the representation the recurrence executes at
 */
public record DeltaRuleUpdate(
        OperandRef query,
        OperandRef key,
        OperandRef value,
        OperandRef decay,
        OperandRef beta,
        OperandRef state,
        OperandRef output,
        int valueHeads,
        int keyHeads,
        int stateDim,
        DataType dataType)
        implements Operation {

    public DeltaRuleUpdate {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(decay, "decay");
        Objects.requireNonNull(beta, "beta");
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (keyHeads < 1 || valueHeads < 1) {
            throw new IllegalArgumentException(
                    "head counts must be positive: " + valueHeads + " value, " + keyHeads + " key");
        }
        if (valueHeads % keyHeads != 0) {
            throw new IllegalArgumentException(
                    valueHeads
                            + " value heads do not divide evenly among "
                            + keyHeads
                            + " key heads");
        }
        if (stateDim < 1) {
            throw new IllegalArgumentException("stateDim must be at least 1: " + stateDim);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.DELTA_RULE_UPDATE;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(query, key, value, decay, beta, state);
    }

    /** The state is written as well as read: the recurrence is the point. */
    @Override
    public List<OperandRef> outputs() {
        return List.of(output, state);
    }
}
