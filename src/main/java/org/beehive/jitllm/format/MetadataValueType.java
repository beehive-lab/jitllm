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
package org.beehive.jitllm.format;

public enum MetadataValueType {
    // The value is a 8-bit unsigned integer.
    UINT8(1),
    // The value is a 8-bit signed integer.
    INT8(1),
    // The value is a 16-bit unsigned little-endian integer.
    UINT16(2),
    // The value is a 16-bit signed little-endian integer.
    INT16(2),
    // The value is a 32-bit unsigned little-endian integer.
    UINT32(4),
    // The value is a 32-bit signed little-endian integer.
    INT32(4),
    // The value is a 32-bit IEEE754 floating point number.
    FLOAT32(4),
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the
    // reader being buggy.
    BOOL(1),
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    STRING(-8),
    // The value is an array of other values, with the length and type prepended.
    // Arrays can be nested, and the length of the array is the number of elements in the array, not
    // the number of bytes.
    ARRAY(-8),
    // The value is a 64-bit unsigned little-endian integer.
    UINT64(8),
    // The value is a 64-bit signed little-endian integer.
    INT64(8),
    // The value is a 64-bit IEEE754 floating point number.
    FLOAT64(8);
    private final int byteSize;

    MetadataValueType(int byteSize) {
        this.byteSize = byteSize;
    }

    private static final MetadataValueType[] VALUES = values();

    public static MetadataValueType fromIndex(int index) {
        return VALUES[index];
    }

    public int byteSize() {
        return byteSize;
    }
}
