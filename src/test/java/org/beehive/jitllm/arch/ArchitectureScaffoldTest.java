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
