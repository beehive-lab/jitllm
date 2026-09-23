package org.beehive.jllm.format;

import java.lang.foreign.MemorySegment;

public record GGMLTensorEntry(
        MemorySegment mappedFile,
        String name,
        GGMLType ggmlType,
        int[] shape,
        MemorySegment memorySegment) {}
