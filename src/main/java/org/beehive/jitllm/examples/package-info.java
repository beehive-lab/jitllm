/**
 * Runnable examples of the public API.
 *
 * <p>Each class here is a {@code main} that does one thing and prints what it did. They are the
 * worked answer to "how do I use this from my own code": they import only {@code api} types, they
 * never touch a backend, a task graph or a GGUF type, and they are the same shape an embedder's
 * code should be.
 *
 * <p>Run one with the launcher's classpath, passing a model file:
 *
 * <pre>
 *   java -cp target/jitllm-1.0.0-jdk21.jar \
 *        org.beehive.jitllm.examples.HelloGeneration model.gguf
 * </pre>
 *
 * <p>To run on an accelerator, launch through {@code jitllm}, which sets the JVM flags and {@code
 * -Duse.tornadovm=true} that {@link org.beehive.jitllm.api.ModelOptions} reads.
 */
package org.beehive.jitllm.examples;
