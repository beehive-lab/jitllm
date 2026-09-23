package org.beehive.jllm.backend.tornado.plan;

import java.util.Arrays;
import java.util.List;
import org.beehive.jllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Which combinations of operand representations a fused device kernel accepts.
 *
 * <p>Mixed quantization between tensors is legal and normal — Qwen3.8-27B holds Q4_0 projections,
 * Q4_1 down projections, Q5_K recurrent outputs and a Q6_K vocabulary projection. What is not legal
 * is handing a <b>fused</b> kernel a combination it was not written for. A kernel that decodes
 * three operands as Q4_0 blocks reads 18 bytes per 32 weights; give it a Q5_K operand and it reads
 * that tensor's 176-byte super-blocks as ten Q4_0 blocks. Every weight it produces is of plausible
 * magnitude and the model generates fluent, wrong text.
 *
 * <p>So a fused kernel states the tuples it supports, and anything else is refused <b>here</b>, at
 * plan construction, with the operands named. Not silently converted, and not discovered later.
 *
 * <p>Today every fused kernel in this backend requires its quantized operands to share one
 * representation, which is why {@link #requireUniform} is the only rule. A future kernel that
 * genuinely handles a mixed tuple would declare that tuple rather than loosen this.
 */
public final class FusedOperandSupport {

    private FusedOperandSupport() {}

    /**
     * Requires every operand of a fused task to share one representation.
     *
     * @param kernel the fused kernel's name, for the message
     * @param operandNames what each operand is, in order — a role, not a buffer
     * @param operands the tensors that would be bound, in the same order
     * @throws UnsupportedOperationException naming the kernel, the operands and their
     *     representations, when they do not agree
     */
    public static void requireUniform(
            String kernel, List<String> operandNames, TornadoTensor... operands) {
        if (operands.length == 0) {
            return;
        }
        DataType first = operands[0].dataType();
        boolean uniform = true;
        for (TornadoTensor operand : operands) {
            uniform &= operand.dataType() == first;
        }
        if (uniform) {
            return;
        }
        StringBuilder detail = new StringBuilder();
        for (int i = 0; i < operands.length; i++) {
            if (i > 0) {
                detail.append(", ");
            }
            detail.append(i < operandNames.size() ? operandNames.get(i) : ("operand " + i))
                    .append('=')
                    .append(operands[i].dataType());
        }
        throw new UnsupportedOperationException(
                kernel
                        + " decodes all of its weight operands with one block layout, and was given"
                        + " a mixture: "
                        + detail
                        + ". Reading one layout as another produces weights of plausible magnitude"
                        + " and fluent, wrong output, so this is refused rather than converted."
                        + " Either split the fused task into per-tensor ones, or add a kernel that"
                        + " states this combination.");
    }

    /** Whether these operands share one representation. Query form of {@link #requireUniform}. */
    public static boolean isUniform(TornadoTensor... operands) {
        if (operands.length == 0) {
            return true;
        }
        DataType first = operands[0].dataType();
        return Arrays.stream(operands).allMatch(operand -> operand.dataType() == first);
    }
}
