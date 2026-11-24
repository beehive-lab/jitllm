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
/**
 * The operation vocabulary — what transformer inference does, named once and independently of model
 * family.
 *
 * <p>Today the same ten or so pieces of work are written twice: once as plain Java in {@code
 * inference.InferenceCore}, and once as TornadoVM kernels and task-graph entries under {@code
 * tornadovm.kernels} and {@code tornadovm.layers}. Neither names the work in a form the other can
 * read, so an architecture is implemented once per backend. This package is the shared name for it.
 * Unifying the <b>vocabulary</b> does not mean unifying the implementations ({@code
 * target-architecture.md}, "Reusable operations").
 *
 * <h2>What an operation is here</h2>
 *
 * <p><b>A description, not a call.</b> An {@link org.beehive.jitllm.program.op.Operation} says
 * which work is done, over which operands, with which configuration. It does not execute, does not
 * hold buffers, and does not know a backend. That is what lets one Llama description be compiled by
 * the TornadoVM backend and executed by the CPU one.
 *
 * <ul>
 *   <li><b>No {@code DataType} parameterization.</b> It arrives at the description and
 *       dispatch level — not inside kernel bodies, because TornadoVM compiles per concrete native
 *       array type and Java has no generics over primitives.
 *   <li><b>No implementations and no dispatch.</b> The CPU forward passes are expressed in these
 *   <li><b>No operator registry, graph optimizer or fusion rules.</b> Explicit non-goals ({@code
 *   <li><b>No {@code Dequantize} operation.</b> Dequantization is a materialization concern. A
 *   <li><b>No graph.</b> Programs are ordered component lists, not a graph IR.
 *       These types carry no edges and no successors.
 * </ul>
 *
 * <h2>Rules this package must keep passing</h2>
 *
 * <ul>
 *   <li><b>Rule 3</b> — nothing here may import TornadoVM or a backend implementation. The rule was
 *   <li><b>Rule 4</b> — no GGUF or GGML type reaches an operation signature.
 *   <li><b>Rule 14</b> — no tokenizer, chat format, sampler policy or generation loop. An operation
 *       vocabulary that assumed token generation could not serve embeddings, classification or
 *       reranking. Note {@link org.beehive.jitllm.program.op.Sample} and {@link
 *       org.beehive.jitllm.program.op.ArgMax}: sampling <i>is</i> an operation and may execute
 *       on the device (Rule 8b). What Rule 14 forbids is <i>requiring</i> one.
 * </ul>
 */
package org.beehive.jitllm.program.op;
