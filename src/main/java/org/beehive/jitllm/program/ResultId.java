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
package org.beehive.jitllm.program;

/**
 * Something the caller may read once an invocation completes.
 *
 * <p>A result names what may be read and from where; it is not a buffer the caller owns. The array
 * it is read from is a program-fixed binding like any other.
 */
public enum ResultId {

    /** The vocabulary distribution. A tensor result: its representation comes from its carrier. */
    LOGITS,

    /** The chosen token, when sampling ran on the device. A scalar result. */
    SAMPLED_TOKEN
}
