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
package org.beehive.jitllm.backend.tornado.lowering;

/**
 * A lowering's refusal, naming what it expected and what it was handed.
 *
 * <p>Top-level rather than nested in one family's lowering, because there are two families now and
 * a refusal is the same event whichever raised it. What differs is the {@code family} prefix, which
 * is why it is a parameter rather than a constant in the message.
 *
 * <p>The message names <b>both sides</b> on purpose. "unsupported program" tells whoever reads the
 * log nothing they can act on; "layer_0 component 4 rotary layout expected NEOX but found
 * INTERLEAVED" tells them which description to look at and which line of it is wrong.
 */
public final class UnsupportedProgramException extends RuntimeException {

    UnsupportedProgramException(String family, String what, String expected, String found) {
        super(
                "this lowering handles "
                        + family
                        + " only; "
                        + what
                        + " expected "
                        + expected
                        + " but found "
                        + found);
    }
}
