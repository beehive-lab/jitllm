package org.beehive.jllm.runtime.backend;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.EnumSet;
import org.junit.Test;

/**
 * There is little behaviour to test, because there is deliberately little behaviour: the task is
 * the declaration. What is worth pinning is the taxonomy itself, since a fourth constant appearing
 * would be a change to the ownership model that should not pass unnoticed.
 */
public class BufferLifetimeTest {

    @Test
    public void thereAreExactlyThreeLifetimesAndTheyAreTheOnesTheMatrixNames() {
        assertEquals(
                EnumSet.of(BufferLifetime.MODEL, BufferLifetime.ENGINE, BufferLifetime.INVOCATION),
                EnumSet.allOf(BufferLifetime.class));
    }

    @Test
    public void theTaxonomyIsClosedUnlikeBackendsAndCapabilities() {
        // BackendId and DeviceCapability are open values on purpose: one can arrive without this
        // project knowing in advance. A fourth buffer lifetime is a change to the ownership model,
        // so it must require editing the enum and the matrix together.
        assertEquals(3, BufferLifetime.values().length);
        assertTrue(
                "an enum, so the set cannot be extended from outside",
                BufferLifetime.class.isEnum());
    }
}
