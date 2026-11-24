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
package org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.nio.file.Path;
import java.util.Optional;
import org.junit.Test;

/**
 * Whether the cuBLAS the loader resolves is new enough for the GPU, from the library's file name.
 */
public class CublasVersionCheckTest {

    @Test
    public void versionComesFromTheFileName() {
        assertEquals(1200, NativePrefillSupport.cublasVersionOf("libcublasLt.so.12.0.2.224"));
        assertEquals(1208, NativePrefillSupport.cublasVersionOf("libcublasLt.so.12.8.4.1"));
        assertEquals(1301, NativePrefillSupport.cublasVersionOf("libcublasLt.so.13.1.1.3"));
        assertEquals(1300, NativePrefillSupport.cublasVersionOf("libcublasLt.so.13"));
        assertEquals(-1, NativePrefillSupport.cublasVersionOf("libcublasLt.so"));
    }

    @Test
    public void cublas12_0IsTooOldForBlackwell() {
        Optional<String> reason =
                NativePrefillSupport.cublasTooOld(
                        120, Path.of("/lib/x86_64-linux-gnu/libcublasLt.so.12.0.2.224"));
        assertTrue(reason.isPresent());
        assertTrue(reason.get(), reason.get().contains("cuBLAS 12.0"));
        assertTrue(
                reason.get(), reason.get().contains("compute capability 12.0 needs cuBLAS 12.8"));
    }

    @Test
    public void cublas12_8AndNewerServeBlackwell() {
        assertFalse(
                NativePrefillSupport.cublasTooOld(120, Path.of("/x/libcublasLt.so.12.8.4.1"))
                        .isPresent());
        assertFalse(
                NativePrefillSupport.cublasTooOld(100, Path.of("/x/libcublasLt.so.13.1.1.3"))
                        .isPresent());
    }

    @Test
    public void olderGpusAndUnversionedFilesAreNotJudged() {
        assertFalse(
                NativePrefillSupport.cublasTooOld(80, Path.of("/x/libcublasLt.so.12.0.2.224"))
                        .isPresent());
        assertFalse(
                NativePrefillSupport.cublasTooOld(90, Path.of("/x/libcublasLt.so.12.0.2.224"))
                        .isPresent());
        assertFalse(
                NativePrefillSupport.cublasTooOld(120, Path.of("/x/libcublasLt.so")).isPresent());
        assertFalse(NativePrefillSupport.cublasTooOld(-1).isPresent());
    }
}
