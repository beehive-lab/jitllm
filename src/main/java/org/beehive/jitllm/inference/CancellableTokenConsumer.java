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

import java.util.function.IntConsumer;

/**
 * A token consumer that can also ask the generation loop to stop. The loops in {@link
 * TokenGenerationLoop} check it after each token they deliver, and leave the loop the same way a
 * stop token does, so the caller's end-of-turn bookkeeping runs as usual.
 */
public interface CancellableTokenConsumer extends IntConsumer {

    /** Whether generation should stop after the token just delivered. */
    boolean cancellationRequested();
}
