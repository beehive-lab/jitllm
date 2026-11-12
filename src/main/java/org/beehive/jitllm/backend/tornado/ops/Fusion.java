package org.beehive.jitllm.backend.tornado.ops;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.program.op.Operation;
import org.beehive.jitllm.program.op.OperationKind;

/**
 * A sequence of program operations that an implementation runs as one step, for example RMS norm
 * followed by the q, k and v projections. One entry of an implementation's fusion table, as in
 * llama.cpp's Metal backend ({@code ggml-metal-fusion.cpp}).
 *
 * @param name a short name for messages, for example {@code "norm + q|k|v"}
 * @param kinds the operation kinds, in order
 * @param check an extra condition on the matched operations, for example "the projections read the
 *     same input"
 */
public record Fusion(String name, List<OperationKind> kinds, Predicate<List<Operation>> check) {

    public Fusion {
        kinds = List.copyOf(kinds);
        if (kinds.isEmpty()) {
            throw new IllegalArgumentException("a fusion needs at least one operation");
        }
    }

    /** The fusion of {@code kinds}, with no extra condition. */
    public static Fusion of(String name, OperationKind... kinds) {
        return new Fusion(name, List.of(kinds), ops -> true);
    }

    /** This fusion, matching only when {@code condition} also holds. */
    public Fusion when(Predicate<List<Operation>> condition) {
        return new Fusion(name, kinds, check.and(condition));
    }

    /** The operations of {@code ops} starting at {@code at} if this fusion matches them. */
    public Optional<List<Operation>> match(List<Operation> ops, int at) {
        if (at + kinds.size() > ops.size()) {
            return Optional.empty();
        }
        List<Operation> candidate = ops.subList(at, at + kinds.size());
        for (int i = 0; i < kinds.size(); i++) {
            if (candidate.get(i).kind() != kinds.get(i)) {
                return Optional.empty();
            }
        }
        return check.test(candidate) ? Optional.of(List.copyOf(candidate)) : Optional.empty();
    }

    /** Whether every projection in {@code ops} reads the same input, as q, k and v do. */
    public static boolean sameInput(List<Operation> ops) {
        List<MatMul> matMuls = new ArrayList<>();
        for (Operation op : ops) {
            if (op instanceof MatMul m) {
                matMuls.add(m);
            }
        }
        for (MatMul m : matMuls) {
            if (!m.input().equals(matMuls.get(0).input())) {
                return false;
            }
        }
        return true;
    }
}
