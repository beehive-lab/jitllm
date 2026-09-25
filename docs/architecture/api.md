# Public API

The facade is the only supported way in. Abstracts away the runtime and accelerator components.
Hence, it exposes no TornadoVM type, no GGUF type and no CLI type in any signature.

## Stability

Thirteen types are **stable**. They may gain members; they do not change or lose them
without a major version.

`LocalModels` · `LocalModel` · `TextGenerationModel` · `GenerationSession` ·
`GenerationRequest` · `GenerationResult` · `GenerationEvent` · `ModelOptions` ·
`SessionOptions` · `ChatRole` · `ChatContent` · `ToolSpec` · `ThinkingMode`

Everything else that is reachable from the facade carries `@Experimental`, including types
that live outside `api/**` — `BackendId`, `DeviceSelector`, `ExecutionPolicy`,
`StorageOptions`, `MemoryPlan`, `DataType`, `DiagnosticCode` — because living outside the
package does not make a type stable. `@Experimental` permits breaking changes; in exchange
each one must be documented, and bridged where a bridge can be honest. A bridge that
quietly changes behaviour is worse than a compile error.

`StableSurfaceTest` enforces both halves: a public `api/**` type in neither list is an
undeclared decision and fails the build, and a stable member may not expose an experimental
type in its signature.

## Shape

```java
try (LocalModel model = LocalModels.load(path, ModelOptions.defaults())) {
    TextGenerationModel text = (TextGenerationModel) model;
    try (GenerationSession session = text.newSession()) {
        GenerationResult result = session.generate(
            GenerationRequest.builder()
                .prompt("What is the capital of France?")
                .maxNewTokens(64)
                .onEvent(e -> System.out.print(e.text()))
                .build());
    }
}
```

`LocalModel` is the loaded model and nothing more: identity, configuration, close. There is
no `forward(...)`, no sampler accessor and no plan accessor on it — execution belongs below
the model and generation policy above it. Generation is a **capability**:
`TextGenerationModel` is where `newSession()` is declared, so a model that only produces
embeddings is a `LocalModel` and nothing else.

## Options

`ModelOptions` is fixed for the model's life; `SessionOptions` refines a session within it.

| `ModelOptions` | |
| --- | --- |
| `contextLength` | sequence budget |
| `backend` | `BackendId`, or resolved from `-Duse.tornadovm` |
| `device` | a `DeviceSelector`; a selector this build cannot honour throws rather than falling back |
| `executionPolicy` | phase strategy, prefill batch size, sampling residency, attention options |
| `storageOptions` | KV dtype (FP16 by default; `StorageOptions.fp32()` for FP32) and paging |
| `thinkingMode` | default for sessions of this model |
| `maxConcurrentSessions` | how many sessions may be open at once (default 1) |

`maxConcurrentSessions` is a memory decision, like `contextLength`. On the GPU, a model whose
sessions share one key/value pool reserves it at load for that many sessions at the full load-time
context, plus one scratch block; a session that asks for a shorter `contextLength` does not shrink
the reservation. Otherwise each open session allocates its own cache. Opening one session more than
the limit throws `IllegalStateException` carrying `GPUL-MEM-003`; closing a session frees its place.
Callers that hold several sessions of one model at once must set it. `LocalModels.preflight` and
the CLI's `-v` report size the key/value cache accordingly.

An FP16 cache the loaded model's kernels do not implement is refused with
`IllegalArgumentException` (`GPUL-CFG-002`) when the model loads, or when a session's execution
policy override selects such a path, before any cache is allocated. The message names the
combination and the FP32 setting. See [kv-cache-support.md](kv-cache-support.md).

| `SessionOptions` | |
| --- | --- |
| `contextLength` | per-session budget |
| `executionPolicy` | *overrides* onto the model's policy, not a replacement |
| `thinkingMode` | per-session |

Execution policy is resolved once per generation, never per token.

## Requests and results

`GenerationRequest` carries either a `prompt` (with an optional `systemPrompt`) or a full
`messages` list — the whole conversation, not just the latest turn — plus sampling
parameters, stop sequences, `tools`, and the streaming callbacks.

`GenerationResult` carries the text, prompt and generated token counts, a `FinishReason`,
timings and any tool calls.

`FinishReason` is one of `STOP_TOKEN`, `MAX_TOKENS`, `STOP_SEQUENCE`, `CONTEXT_FULL`,
`TOOL_CALL`.

## Conversations

`ChatMessage` pairs a `ChatRole` (`SYSTEM`, `USER`, `ASSISTANT`, `TOOL`) with `ChatContent`,
which is one of `Text`, `ToolCall` or `ToolResult`. The facade renders the family's chat
template itself, including tool definitions: a caller passes structured messages and never
assembles a template.

Passing `messages` supplies the whole conversation. A session also retains its own history
across `generate` calls; `reset()` starts the conversation over without releasing the
session's storage.

## Tools

`ToolSpec` describes a callable tool. A model that emits a tool call returns
`FinishReason.TOOL_CALL` and the calls on `GenerationResult.toolCalls()`; the caller
executes them and sends the results back as `ChatContent.ToolResult` messages.

Tool calling does not imply forced or named tool choice, and does not imply structured
output. Those are separate capabilities and are not claimed.

A request with tools on a family whose format cannot call them fails with
`[GPUL-REQ-003]`. Ask first instead: `model.info().capabilities().toolCalling()`.

## Thinking control

`ThinkingMode` is `DEFAULT`, `ENABLED` or `DISABLED`. An **explicit** mode on a family with
no reasoning phase is rejected rather than ignored, so a caller who asks for something the
model cannot do is told. `DEFAULT` leaves it to the family. Ask first with
`model.info().capabilities().thinkingControl()`.

## Capabilities

`ModelInfo.capabilities()` (experimental) returns a `ModelCapabilities` record,
`(toolCalling, thinkingControl)`, read from the loaded model's chat format. It is the same
predicate the request path enforces, so an integration can refuse tools or thinking control
at startup with its own message instead of catching the exception from the first request.

## Streaming

`onEvent` receives one ordered `GenerationEvent` per emitted token, carrying the token id
and the text that token completed — possibly empty, because UTF-8 is decoded incrementally
and a multi-byte character is never split across events. Concatenating every event's text
equals `GenerationResult.text()`, subject only to stop-sequence truncation, which is applied
to the finished string.

Terminal stop and control tokens are not emitted and are not counted in
`generatedTokens()`.

The callback runs **outside** the invocation lock: results are copied out, the lock is
released, and only then does caller code run. Holding the lock across arbitrary user code
would let one caller stall every other session sharing the compiled program, and
re-entering the session from the callback would deadlock.

## Lifecycle

| Rule | |
| --- | --- |
| Close order | sessions before the model. `LocalModel.close()` throws `GPUL-LIFE-002` while sessions are open, naming them, and never force-closes one |
| Failed close | has no effect — the model stays open and usable, so the caller can close the sessions and retry |
| Idempotence | a successful `close()` on either is a no-op the second time |
| Use after close | throws `IllegalStateException` carrying `GPUL-LIFE-001`; a session also refuses once its *model* is closed |
| After model close | no new session may be opened |
| Session limit | at most `maxConcurrentSessions` open at once; the next `newSession()` throws `GPUL-MEM-003` until one closes |
| Reuse | a session is reusable across generations; its position advances and `reset()` rewinds it |
| Concurrency | a session is **not** thread-safe. A loaded model is |

The natural spelling is nested try-with-resources — model outer, session inner — so the
honest failure surfaces ordering bugs instead of hiding them.

## Errors

Failures carry a `DiagnosticCode` in the message, so a cause is identifiable without
parsing prose: `GPUL-MOD-002` for an unrecognized model, `GPUL-MEM-001` for a load known to
exceed device capacity, `GPUL-LIFE-001` and `GPUL-LIFE-002` for lifecycle misuse.
Device-memory exhaustion surfaces as `InsufficientDeviceMemoryException` rather than a
backend exception: backend exception types stop at the backend boundary.
