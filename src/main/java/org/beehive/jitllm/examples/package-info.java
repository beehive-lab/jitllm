/**
 * Runnable examples of the public API.
 *
 * <p>Each class here is a {@code main} that does one thing and prints what it did. They are the
 * worked answer to "how do I use this from my own code": they import only {@code api} types, they
 * never touch a backend, a task graph or a GGUF type, and they are the same shape an embedder's
 * code should be.
 *
 * <p>Run one the way the launcher does, with the TornadoVM SDK's argument file (the jar carries
 * jitllm's classes only; TornadoVM comes from the SDK), passing a model file:
 *
 * <pre>
 *   java @$TORNADOVM_HOME/tornado-argfile --add-modules jdk.incubator.vector \
 *        -cp target/jitllm-&lt;version&gt;.jar \
 *        org.beehive.jitllm.examples.HelloGeneration model.gguf
 * </pre>
 *
 * <p>To run on an accelerator, launch through {@code jitllm}, which sets the JVM flags and {@code
 * -Duse.tornadovm=true} that {@link org.beehive.jitllm.api.ModelOptions} reads.
 */
package org.beehive.jitllm.examples;
