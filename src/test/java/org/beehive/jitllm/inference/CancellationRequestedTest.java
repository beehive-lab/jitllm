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
