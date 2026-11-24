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
package org.beehive.jitllm.arch;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.tngtech.archunit.core.domain.JavaClasses;
import org.junit.Test;

public class ArchitectureScaffoldTest {

    @Test
    public void importsTheProductionTree() {
        JavaClasses classes = ProductionClasses.get();
        assertFalse("no production classes imported — check the module layout", classes.isEmpty());
        assertTrue("expected the production tree to be substantial", classes.size() > 100);
    }

    @Test
    public void excludesTestClasses() {
        boolean anyTestClass =
                ProductionClasses.get().stream().anyMatch(c -> c.getName().endsWith("Test"));
        assertFalse("test classes must not be part of the rule input", anyTestClass);
    }
}
