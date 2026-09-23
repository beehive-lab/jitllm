package org.beehive.jllm.api;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * This type is not yet stable: it may change or disappear in a future release.
 *
 * <h2>What the marker permits, and what it obliges</h2>
 *
 * <p>While a type carries this annotation, breaking changes to it <b>are permitted</b>, up to and
 * including the milestone that freezes the API. In exchange:
 *
 * <ul>
 *   <li>every breaking change <b>must be documented</b> — what changed, and what replaces it;
 *   <li>a <b>deprecation bridge is provided when it is inexpensive and meaningful</b>. It is not
 *       required where the old and new shapes cannot be bridged honestly — a bridge that quietly
 *       changes behaviour is worse than a compile error.
 * </ul>
 *
 * <p><b>Removing this marker freezes the API.</b> That is the point at which these allowances end,
 * so the annotation disappearing from a type is itself the compatibility commitment.
 *
 * <p>The annotation is the project's own rather than a dependency, for the same reason the project
 * ships no logging facade: a self-contained inference library should not acquire a dependency to
 * say one word. It is {@link RetentionPolicy#CLASS} — visible to tools reading the bytecode,
 * without forcing the annotation onto the runtime classpath of anyone who depends on the library.
 */
@Documented
@Retention(RetentionPolicy.CLASS)
@Target({
    ElementType.TYPE,
    ElementType.METHOD,
    ElementType.CONSTRUCTOR,
    ElementType.FIELD,
    ElementType.PACKAGE
})
public @interface Experimental {

    /** Optional note: what is expected to change, or which milestone settles it. */
    String value() default "";
}
