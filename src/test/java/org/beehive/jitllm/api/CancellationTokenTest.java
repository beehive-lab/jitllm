/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
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
