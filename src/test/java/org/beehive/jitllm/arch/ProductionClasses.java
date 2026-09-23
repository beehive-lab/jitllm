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
