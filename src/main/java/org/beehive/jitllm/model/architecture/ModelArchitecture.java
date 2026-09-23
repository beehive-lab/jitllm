package org.beehive.jllm.model.architecture;

import java.util.Set;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.program.InferenceProgram;
import org.beehive.jllm.program.PhaseId;
import org.beehive.jllm.runtime.model.ArchitectureId;

/**
 * What a model computes — one component per architecture, and <b>backend-neutral</b>.
 *
 * <h2>What it owns</h2>
 *
 * <ul>
 *   <li>its {@link ArchitectureId};
 *   <li>validation of the architecture-specific <b>configuration shape</b>;
 *   <li>the logical {@link InferenceProgram} — operations, components, and the <b>logical phases it
 *       can express</b>.
 * </ul>
 *
 * <h2>What it must not do</h2>
 *
 * <p><b>It never declares that an (architecture, dtype, mode) triple is executable.</b> That is
 * backend capability, and it lives in the backend's lowering registration. So this interface, and
 * every implementation of it, contains <b>no TornadoVM type, no device capability, no backend
 * identifier, no task graph, no kernel, and no support matrix</b> — Rule 3, and the reason the
 * descriptions moved out of {@code tornadovm.lowering} to get here.
 *
 * <p><b>A description existing does not imply a lowering exists.</b> Four of the ten architectures
 * this project loads have no description at all, and two that do have no backend support for some
 * of their dtype and mode combinations. Those are separate facts, held in separate places, and
 * collapsing them would make "we can describe it" read as "we can run it".
 *
 * <p><b>Logical phases are not execution modes.</b> {@link #logicalPhases()} says what the program
 * can express — most architectures can express prefill and decode. Whether a backend has a
 * prefill/decode plan for a given dtype is registered, not described: Mistral and Granite express
 * both phases today and their Tornado registration supports only {@code STANDARD}.
 *
 * <h2>Discovery</h2>
 *
 * <p>Through {@link java.util.ServiceLoader}, like {@code ModelProvider}. Adding an architecture is
 * adding a file and a service line — never editing a list. {@link ModelArchitectures} makes the two
 * failure modes deterministic: <b>duplicate identities name both implementations</b>, and a
 * <b>missing one fails by identity</b>.
 *
 * <p>Recognition and loading stay a separate SPI ({@code ModelProvider}): one answers "what is this
 * file", this one answers "what does this model compute".
 */
public interface ModelArchitecture {

    /** Which architecture this computes. Two implementations claiming one identity is an error. */
    ArchitectureId id();

    /**
     * Checks that this configuration is the shape this architecture needs.
     *
     * @throws IllegalArgumentException naming the field that is missing or inconsistent
     */
    void validateConfiguration(Configuration configuration);

    /**
     * The logical phases this architecture's program can express.
     *
     * <p><b>Not backend execution modes.</b> See the class javadoc: a backend registers what it can
     * run, and that set is usually smaller.
     */
    Set<PhaseId> logicalPhases();

    /** Describes the forward pass as a program. */
    InferenceProgram describe(ArchitectureInputs inputs);
}
