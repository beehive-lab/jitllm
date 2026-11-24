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
 * Backend-neutral identities, selectors, options and contracts.
 *
 * <p>Everything above the backend — {@code program}, {@code api}, {@code engine}, the runtime and
 * the compiled-program cache keys — may depend on this package. <b>Nothing may depend on {@code
 * backend.cpu} or {@code backend.tornado} directly.</b> That is the whole arrangement: the upper
 * layers name what they want, and an implementation package answers.
 *
 * <p><b>This package stays small.</b> It is not a home for implementation utilities. A type belongs
 * here only if a layer above the backend must name it; anything a backend merely finds convenient
 * belongs in that backend.
 *
 * <p>It must not import TornadoVM [Rule 1], and it must not import an implementation package — both
 * are asserted in {@code DependencyRulesTest}.
 */
package org.beehive.jitllm.runtime.backend;
