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

import java.util.Objects;

/**
 * One emitted completion token: its id, and the text it completed.
 *
 * <p><b>One event, not two callbacks</b> [§7]. A separate {@code onToken(String)} and {@code
 * onTokenId(int)} would hand the consumer two streams to reassemble, with nothing in the API
 * guaranteeing that the <i>n</i>-th id belongs with the <i>n</i>-th string — and the one consumer
 * that needs both, a streaming chat model, is exactly the one that would have to trust it.
 *
 * <h2>{@code text} may be empty, and that is not a gap</h2>
 *
 * <p>A token can carry part of a multi-byte character: the id is real, and the text is not there
 * yet. When the bytes complete, the text is attached to <b>the event that completes it</b>. Text is
 * also empty for a token the tokenizer does not display.
 *
 * <p>The invariant a consumer can rely on: <b>concatenating every non-empty {@code text} in order
 * equals {@link GenerationResult#text()}</b>, subject only to stop-sequence truncation, which is
 * applied to the finished string after generation.
 *
 * @param tokenId the model's token id — never a terminal stop or control token, which are not
 *     emitted and are not counted in {@link GenerationResult#generatedTokens()}
 * @param text the text this token completed, possibly empty
 */
public record GenerationEvent(int tokenId, String text) {

    public GenerationEvent {
        Objects.requireNonNull(text, "text; use an empty string for a token that completes none");
    }

    /** Whether this event carries text a display would show. */
    public boolean hasText() {
        return !text.isEmpty();
    }
}
