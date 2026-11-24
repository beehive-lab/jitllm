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
package org.beehive.jitllm.quality;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import org.junit.Test;

/** The scoring arithmetic and the alignment, without a model in sight. */
public class NllScoringTest {

    private static final double TOLERANCE = 1e-12;

    @Test
    public void logSumExpMatchesTheClosedFormOnAUniformRow() {
        float[] logits = {2.0f, 2.0f, 2.0f, 2.0f};
        // log(4 * e^2) = 2 + log 4
        assertEquals(2.0 + Math.log(4), NllScoring.logSumExp(logits), TOLERANCE);
    }

    @Test
    public void logSumExpSurvivesLogitsThatWouldOverflowAFloat() {
        float[] logits = {200.0f, 199.0f};
        assertEquals(200.0 + Math.log(1 + Math.exp(-1)), NllScoring.logSumExp(logits), 1e-9);
    }

    @Test
    public void aUniformRowCostsLogOfTheVocabulary() {
        float[] logits = new float[8];
        assertEquals(Math.log(8), NllScoring.negativeLogLikelihood(logits, 3), TOLERANCE);
    }

    @Test
    public void aCertainPredictionCostsNothing() {
        float[] logits = {0.0f, 100.0f, 0.0f};
        assertEquals(0.0, NllScoring.negativeLogLikelihood(logits, 1), 1e-9);
    }

    @Test
    public void aNonFiniteLogitIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> NllScoring.negativeLogLikelihood(new float[] {1.0f, Float.NaN}, 0));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        NllScoring.negativeLogLikelihood(
                                new float[] {1.0f, Float.POSITIVE_INFINITY}, 0));
    }

    @Test
    public void aTargetOutsideTheVocabularyIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> NllScoring.negativeLogLikelihood(new float[] {1.0f, 2.0f}, 2));
    }

    /** The alignment: row t predicts token t+1, and the first token is the unscored prefix. */
    @Test
    public void theRowAfterATokenPredictsTheNextOne() {
        int[] tokens = {11, 22, 33, 44};
        assertArrayEquals(
                new int[][] {{0, 22}, {1, 33}, {2, 44}}, NllScoring.scoredPositions(tokens));
        assertEquals(
                "a passage of n tokens scores n-1", 3, NllScoring.scoredPositions(tokens).length);
        assertEquals(0, NllScoring.scoredPositions(new int[] {7}).length);
    }
}
