package org.beehive.jitllm.backend.tornado.ops;

import java.util.List;
import org.beehive.jitllm.program.op.Operation;

/**
 * Consecutive program operations of one layer that a plan runs through implementations, for example
 * a layer's operations before attention: RMS norm, then the q, k and v projections.
 *
 * @param layer the transformer layer
 * @param name a short name for messages, for example {@code "before attention"}
 * @param tokens the tokens of one chunk
 * @param ops the operations, in the order they run
 */
public record Region(int layer, String name, int tokens, List<Operation> ops) {

    public Region {
        ops = List.copyOf(ops);
    }
}
