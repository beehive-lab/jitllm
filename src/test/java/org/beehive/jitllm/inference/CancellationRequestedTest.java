package org.beehive.jitllm.inference;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.IntConsumer;
import org.junit.Test;

/** The check every decode loop in {@link TokenGenerationLoop} makes after delivering a token. */
public class CancellationRequestedTest {

    @Test
    public void aPlainConsumerIsNeverCancelled() {
        IntConsumer plain = token -> {};
        assertFalse(TokenGenerationLoop.cancellationRequested(plain));
    }

    @Test
    public void noConsumerIsNeverCancelled() {
        assertFalse(TokenGenerationLoop.cancellationRequested(null));
    }

    @Test
    public void aCancellableConsumerIsReadEveryTime() {
        AtomicBoolean stop = new AtomicBoolean();
        CancellableTokenConsumer consumer =
                new CancellableTokenConsumer() {
                    @Override
                    public void accept(int token) {}

                    @Override
                    public boolean cancellationRequested() {
                        return stop.get();
                    }
                };
        assertFalse(TokenGenerationLoop.cancellationRequested(consumer));
        stop.set(true);
        assertTrue(TokenGenerationLoop.cancellationRequested(consumer));
    }
}
