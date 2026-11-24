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
package org.beehive.jitllm.program;

import java.util.List;
import java.util.Objects;
import java.util.Set;
import org.beehive.jitllm.program.op.Operation;

/**
 * One named unit of a program's work.
 *
 * <p>A component is either a single {@linkplain Leaf operation} or a {@linkplain Composite named
 * group} of components — a transformer layer being the obvious group. That gives structure without
 * giving a general graph: components are <b>ordered</b>, carry no edges and no successors, and
 * nothing traverses them.
 *
 * <p>Value-comparable, like everything that ends up inside a {@link ProgramSignature}: no arrays,
 * defensively copied collections, structural equality.
 */
public sealed interface ProgramComponent {

    /** This component's name, unique within its program. */
    String name();

    /** The phases this component participates in. */
    Set<PhaseId> phases();

    /** A single operation. */
    record Leaf(String name, Operation operation, Set<PhaseId> phases) implements ProgramComponent {
        public Leaf {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(operation, "operation");
            phases = requirePhases(phases);
        }
    }

    /**
     * A named group — a transformer layer, an attention block.
     *
     * <p>A composite's phase set must cover every phase its children participate in: a child that
     * ran in a phase its parent does not would be unreachable, which is a defect rather than a
     * refinement.
     */
    record Composite(String name, List<ProgramComponent> children, Set<PhaseId> phases)
            implements ProgramComponent {
        public Composite {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(children, "children");
            if (children.isEmpty()) {
                throw new IllegalArgumentException(
                        "a composite component must have children: " + name);
            }
            children = List.copyOf(children);
            phases = requirePhases(phases);
            for (ProgramComponent child : children) {
                if (!phases.containsAll(child.phases())) {
                    throw new IllegalArgumentException(
                            "child "
                                    + child.name()
                                    + " runs in phases its"
                                    + " parent "
                                    + name
                                    + " does not: "
                                    + child.phases()
                                    + " against "
                                    + phases);
                }
            }
        }
    }

    private static Set<PhaseId> requirePhases(Set<PhaseId> phases) {
        Objects.requireNonNull(phases, "phases");
        if (phases.isEmpty()) {
            throw new IllegalArgumentException(
                    "a component must participate in at least one phase");
        }
        return Set.copyOf(phases);
    }
}
