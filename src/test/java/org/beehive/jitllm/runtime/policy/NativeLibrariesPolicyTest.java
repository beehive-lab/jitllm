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
package org.beehive.jitllm.runtime.policy;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.After;
import org.junit.Test;

public class NativeLibrariesPolicyTest {

    private final String current = System.getProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY);
    private final String legacy =
            System.getProperty(ExecutionPolicy.LEGACY_NATIVE_PREFILL_PROPERTY);

    @After
    public void restore() {
        set(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, current);
        set(ExecutionPolicy.LEGACY_NATIVE_PREFILL_PROPERTY, legacy);
    }

    @Test
    public void jitIsTheDefault() {
        set(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, null);
        set(ExecutionPolicy.LEGACY_NATIVE_PREFILL_PROPERTY, null);
        assertFalse(ExecutionPolicy.fromSystemProperties().nativeLibraries());
        assertFalse(ExecutionPolicy.builder().build().nativeLibraries());
    }

    @Test
    public void nativeLibrariesAreAnExplicitRequest() {
        set(ExecutionPolicy.LEGACY_NATIVE_PREFILL_PROPERTY, null);
        set(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, "true");
        assertTrue(ExecutionPolicy.fromSystemProperties().nativeLibraries());
        set(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, null);
        set(ExecutionPolicy.LEGACY_NATIVE_PREFILL_PROPERTY, "true");
        assertTrue(
                "the legacy spelling is honoured when set",
                ExecutionPolicy.fromSystemProperties().nativeLibraries());
    }

    @Test
    public void disagreeingPropertiesAreRefused() {
        set(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, "true");
        set(ExecutionPolicy.LEGACY_NATIVE_PREFILL_PROPERTY, "false");
        assertThrows(IllegalArgumentException.class, ExecutionPolicy::fromSystemProperties);
    }

    @Test
    public void aSessionCanOverrideIt() {
        ExecutionPolicy model = ExecutionPolicy.builder().build();
        ExecutionPolicy session =
                ExecutionPolicy.Overrides.builder().nativeLibraries(true).build().applyTo(model);
        assertTrue(session.nativeLibraries());
        assertFalse("the override is a different policy", session.equals(model));
        assertTrue(ExecutionPolicy.from(session).build().nativeLibraries());
    }

    private static void set(String key, String value) {
        if (value == null) System.clearProperty(key);
        else System.setProperty(key, value);
    }
}
