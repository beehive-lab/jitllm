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
package org.beehive.jitllm.auxiliary.metrics;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.runtime.metrics.MetricKey;
import org.beehive.jitllm.runtime.metrics.MetricsReport;
import org.junit.Test;

public class InMemoryMetricsSinkTest {

    private static final long ONE_SECOND = 1_000_000_000L;

    @Test
    public void aRunsCountersComeBackAsAReport() {
        InMemoryMetricsSink sink = new InMemoryMetricsSink();
        sink.record(MetricKey.MODEL_LOAD_TIME, 2 * ONE_SECOND);
        sink.record(MetricKey.PROMPT_TOKENS, 21);
        sink.record(MetricKey.PREFILL_TIME, ONE_SECOND / 10);
        sink.record(MetricKey.GENERATED_TOKENS, 64);
        sink.record(MetricKey.DECODE_TIME, ONE_SECOND / 2);

        MetricsReport report = sink.report();
        assertEquals(
                Long.valueOf(2 * ONE_SECOND),
                report.value(MetricKey.MODEL_LOAD_TIME).orElseThrow());
        assertEquals(128.0, report.generatedTokensPerSecond().orElseThrow(), 1e-9);
        assertEquals(210.0, report.promptTokensPerSecond().orElseThrow(), 1e-9);
    }

    @Test
    public void keysCombineByTheirOwnRuleSoTheProducerNeedNotKnow() {
        InMemoryMetricsSink sink = new InMemoryMetricsSink();
        sink.record(MetricKey.DEVICE_KERNEL_TIME, 100);
        sink.record(MetricKey.DEVICE_KERNEL_TIME, 40);
        sink.record(MetricKey.DEVICE_MEMORY_USED, 5_000);
        sink.record(MetricKey.DEVICE_MEMORY_USED, 7_000);

        MetricsReport report = sink.report();
        assertEquals(Long.valueOf(140L), report.value(MetricKey.DEVICE_KERNEL_TIME).orElseThrow());
        assertEquals(
                Long.valueOf(7_000L), report.value(MetricKey.DEVICE_MEMORY_USED).orElseThrow());
    }

    @Test
    public void aReportTakenMidRunIsASnapshotNotAView() {
        InMemoryMetricsSink sink = new InMemoryMetricsSink();
        sink.record(MetricKey.GENERATED_TOKENS, 1);
        MetricsReport early = sink.report();
        sink.record(MetricKey.GENERATED_TOKENS, 1);

        assertEquals(Long.valueOf(1L), early.value(MetricKey.GENERATED_TOKENS).orElseThrow());
        assertEquals(
                Long.valueOf(2L), sink.report().value(MetricKey.GENERATED_TOKENS).orElseThrow());
    }

    @Test
    public void clearingStartsTheNextRunFromNothing() {
        InMemoryMetricsSink sink = new InMemoryMetricsSink();
        sink.record(MetricKey.DECODE_TIME, ONE_SECOND);
        sink.clear();
        assertTrue(sink.report().isEmpty());
    }

    @Test
    public void anEnabledSinkSaysSo() {
        assertTrue(new InMemoryMetricsSink().isEnabled());
    }
}
