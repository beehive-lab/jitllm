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

/**
 * Who is speaking in a conversation turn.
 *
 * <p><b>Project-owned, and an enum.</b> The internal {@code ChatFormat.Role} is an open {@code
 * record Role(String name)} carrying six values, three of which are fill-in-the-middle roles;
 * publishing it would put FIM in the public API as a side effect of a chat surface, and would make
 * every family's role vocabulary a compatibility surface.
 *
 * <p>Closed, unlike {@code BackendId} and {@code DeviceCapability}, which are open values because a
 * backend or a capability can arrive from outside this project. A fifth conversational role is a
 * decision, not an extension point.
 */
public enum ChatRole {

    /** Instructions to the model, outside the dialogue. */
    SYSTEM,

    /** The person or application asking. */
    USER,

    /** The model's own turn: text, tool calls, or both. */
    ASSISTANT,

    /** The result of running a tool the assistant asked for. */
    TOOL
}
