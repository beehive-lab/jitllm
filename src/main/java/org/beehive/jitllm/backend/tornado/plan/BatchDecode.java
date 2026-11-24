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
package org.beehive.jitllm.backend.tornado.plan;

import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The shape of a batched decode step: how many sequences, how long each may grow, and the key/value
 * cache they share.
 *
 * @param batchSize sequences decoded together
 * @param decodeContext positions each sequence may reach
 * @param keyCache the key cache, FP32 or FP16
 * @param valueCache the value cache, in the key cache's precision
 * @param seqPositions each sequence's current position
 * @param blockTable each sequence's cache blocks, or {@code null} for one contiguous region each
 * @param blockSize positions per cache block; unused without a block table
 * @param maxBlocksPerSlot block-table entries per sequence; unused without a block table
 */
public record BatchDecode(
        int batchSize,
        int decodeContext,
        Object keyCache,
        Object valueCache,
        IntArray seqPositions,
        IntArray blockTable,
        int blockSize,
        int maxBlocksPerSlot) {}
