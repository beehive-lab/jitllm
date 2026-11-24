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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import java.util.List;
import org.junit.Test;

/**
 * These are the correctness cases. Reuse that is wrong does not fail loudly: it answers a question
 * conditioned on a context the caller did not send, and only when the conversation happens to have
 * been edited.
 */
public class EncodedPrefixTest {

    @Test
    public void nothingRetainedMeansNoReuseRatherThanDivergence() {
        assertNull(new EncodedPrefix().reusableSuffixOf(List.of(1, 2, 3)));
    }

    @Test
    public void anExactPrefixIsReusedAndOnlyTheRemainderIsReturned() {
        EncodedPrefix prefix = new EncodedPrefix();
        prefix.append(List.of(1, 2, 3));
        assertEquals(List.of(4, 5), prefix.reusableSuffixOf(List.of(1, 2, 3, 4, 5)));
    }

    @Test
    public void anIdenticalInputHasNothingLeftToEncode() {
        EncodedPrefix prefix = new EncodedPrefix();
        prefix.append(List.of(1, 2, 3));
        assertEquals(
                "re-sending the same conversation is legitimate, and encodes nothing",
                List.of(),
                prefix.reusableSuffixOf(List.of(1, 2, 3)));
    }

    @Test
    public void aSingleDivergentTokenDefeatsReuse() {
        EncodedPrefix prefix = new EncodedPrefix();
        prefix.append(List.of(1, 2, 3));
        // This is the case that matters: a tool specification changes the encoded system content,
        // so the input diverges early while the message list looks untouched.
        assertNull(prefix.reusableSuffixOf(List.of(1, 9, 3, 4)));
        assertNull(
                "divergence at the very end counts too",
                prefix.reusableSuffixOf(List.of(1, 2, 9, 4)));
    }

    @Test
    public void anInputShorterThanTheRetainedPrefixCannotExtendIt() {
        EncodedPrefix prefix = new EncodedPrefix();
        prefix.append(List.of(1, 2, 3));
        // A caller that dropped an earlier turn lands here, and gets a re-encode.
        assertNull(prefix.reusableSuffixOf(List.of(1, 2)));
    }

    @Test
    public void generatedTokensJoinTheRetainedPrefix() {
        EncodedPrefix prefix = new EncodedPrefix();
        prefix.append(List.of(1, 2)); // prompt
        prefix.append(List.of(3, 4)); // what the model produced
        assertEquals(
                "a later conversation reuses the assistant turn only if it matches exactly",
                List.of(5),
                prefix.reusableSuffixOf(List.of(1, 2, 3, 4, 5)));
        assertNull(
                "an edited assistant turn re-encodes",
                prefix.reusableSuffixOf(List.of(1, 2, 3, 9, 5)));
    }

    @Test
    public void clearingDropsEverything() {
        EncodedPrefix prefix = new EncodedPrefix();
        prefix.append(List.of(1, 2));
        prefix.clear();
        assertEquals(0, prefix.size());
        assertNull(prefix.reusableSuffixOf(List.of(1, 2, 3)));
    }
}
