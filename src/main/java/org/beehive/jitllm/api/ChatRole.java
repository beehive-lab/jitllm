package org.beehive.jllm.api;

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
