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

import com.tngtech.archunit.core.domain.JavaClasses;
import com.tngtech.archunit.core.importer.ClassFileImporter;
import com.tngtech.archunit.core.importer.ImportOption;

/**
 * Shared import of the production classes the architecture rules run against.
 *
 * <p>Importing is the expensive part of an ArchUnit run, so every rule class reuses this single
 * snapshot. Test classes are excluded: the rules describe the shipped library, and test code is
 * allowed to reach anywhere.
 *
 * <p>Rules and their allowlists are specified in {@code docs/architecture/architecture.md}.
 */
public final class ProductionClasses {

    /** Root package of the shipped library. */
    public static final String ROOT_PACKAGE = "org.beehive.jitllm";

    private static final JavaClasses CLASSES =
            new ClassFileImporter()
                    .withImportOption(ImportOption.Predefined.DO_NOT_INCLUDE_TESTS)
                    .withImportOption(ImportOption.Predefined.DO_NOT_INCLUDE_JARS)
                    .importPackages(ROOT_PACKAGE);

    private ProductionClasses() {}

    public static JavaClasses get() {
        return CLASSES;
    }
}
