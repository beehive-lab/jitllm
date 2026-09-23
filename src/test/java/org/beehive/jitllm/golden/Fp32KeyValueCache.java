package org.beehive.jitllm.golden;

import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.junit.rules.ExternalResource;

/**
 * Pins the FP32 key/value cache for a test class.
 *
 * <p>For tests whose reference — a committed golden, the CPU path, another execution mode — was
 * captured with an FP32 cache and that measure something other than cache precision. FP16 is the
 * default, and its rounding would otherwise be charged to whatever they compare. FP16 accuracy is
 * measured separately, against FP32, by {@link KvPrecisionHarness}.
 *
 * <pre>
 *   &#64;ClassRule public static final Fp32KeyValueCache FP32 = new Fp32KeyValueCache();
 * </pre>
 */
public final class Fp32KeyValueCache extends ExternalResource {

    private String previous;

    @Override
    protected void before() {
        previous = System.getProperty(StorageOptions.FP32_PROPERTY);
        System.setProperty(StorageOptions.FP32_PROPERTY, "true");
    }

    @Override
    protected void after() {
        if (previous == null) {
            System.clearProperty(StorageOptions.FP32_PROPERTY);
        } else {
            System.setProperty(StorageOptions.FP32_PROPERTY, previous);
        }
    }
}
