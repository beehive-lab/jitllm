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
package org.beehive.jitllm.runtime.metrics;

/**
 * The sink that discards everything, reached through {@link MetricsSink#disabled()}.
 *
 * <p>Package-private and stateless on purpose: it exists so that a producer never has to null-check
 * a sink, and so that "metrics off" is an ordinary object rather than a branch threaded through
 * every call site.
 */
final class NoOpMetricsSink implements MetricsSink {

    static final NoOpMetricsSink INSTANCE = new NoOpMetricsSink();

    private NoOpMetricsSink() {}

    @Override
    public void record(MetricKey key, long value) {
        // Discarded by design.
    }

    @Override
    public boolean isEnabled() {
        return false;
    }

    @Override
    public String toString() {
        return "MetricsSink.disabled()";
    }
}
