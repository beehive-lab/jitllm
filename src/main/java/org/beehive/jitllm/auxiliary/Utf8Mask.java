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
package org.beehive.jitllm.auxiliary;

/** mask of a byte-sequence in UTF-8 encoding */
public record Utf8Mask(int mask, int pattern, int len) {
    // @formatter:off
    public static final Utf8Mask[] MASKS = {
        new Utf8Mask(0b11100000, 0b11000000, 2),
        new Utf8Mask(0b11110000, 0b11100000, 3),
        new Utf8Mask(0b11111000, 0b11110000, 4)
    };
    // @formatter:on
}
