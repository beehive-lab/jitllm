package org.beehive.jitllm.api;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** {@link CancellationToken} and its place on {@link GenerationRequest}. No model is needed. */
public class CancellationTokenTest {

    @Test
    public void aNewTokenIsNotCancelled() {
        assertFalse(new CancellationToken().isCancelled());
    }

    @Test
    public void cancelIsSticky() {
        CancellationToken token = new CancellationToken();
        token.cancel();
        assertTrue(token.isCancelled());
        token.cancel();
        assertTrue("cancel() is idempotent", token.isCancelled());
    }

    @Test
    public void cancelIsVisibleFromAnotherThread() throws InterruptedException {
        CancellationToken token = new CancellationToken();
        Thread canceller = new Thread(token::cancel);
        canceller.start();
        canceller.join();
        assertTrue(token.isCancelled());
    }

    @Test
    public void theRequestCarriesItsToken() {
        CancellationToken token = new CancellationToken();
        GenerationRequest request =
                GenerationRequest.builder().prompt("hello").cancellation(token).build();
        assertSame(token, request.cancellation());
    }

    @Test
    public void aRequestWithoutATokenCannotBeCancelled() {
        assertNull(GenerationRequest.of("hello").cancellation());
    }
}
