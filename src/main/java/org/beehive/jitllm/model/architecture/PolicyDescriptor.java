package org.beehive.jllm.model.architecture;

/**
 * The one place an execution policy becomes the string a program signature carries.
 *
 * <p><b>Central by design.</b> Two callers spelling the same policy differently would produce two
 * cache entries for one program, and the symptom — a recompile that looks like a cache which does
 * not work — sits a long way from the cause. One producer makes that unrepresentable.
 *
 * <p>Still a placeholder with a named replacement: the signature is to hold the real {@code
 * ExecutionPolicy} value, at which point this class goes away.
 */
public final class PolicyDescriptor {

    private PolicyDescriptor() {}

    /**
     * The canonical descriptor for a single-token execution policy.
     *
     * @param deviceSample whether sampling is device-resident
     * @param splitKvAttention whether the split-KV attention path is used
     */
    public static String singleToken(boolean deviceSample, boolean splitKvAttention) {
        return "single-token"
                + ";sample="
                + (deviceSample ? "device" : "host")
                + ";splitKv="
                + (splitKvAttention ? "on" : "off");
    }
}
