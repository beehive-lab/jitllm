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
package org.beehive.jllm.runtime.backend;
