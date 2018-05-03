<?php

namespace Rubix\Engine\Metrics;

use InvalidArgumentException;

class F1Score implements Classification
{
    /**
     * Test the F1 score of the predictions.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $outcomes) : float
    {
        if (count($predictions) !== count($outcomes)) {
            throw new InvalidArgumentException('The number of outcomes must match the number of predictions.');
        }

        $labels = array_unique(array_merge($predictions, $outcomes));

        $truePositives = $falsePositives = $falseNegatives = [];

        foreach ($labels as $label) {
            $truePositives[$label] = $falsePositives[$label] = $falseNegatives[$label] = 0;
        }

        foreach ($predictions as $i => $prediction) {
            if ($prediction === $outcomes[$i]) {
                $truePositives[$prediction]++;
            } else {
                $falsePositives[$prediction]++;
                $falseNegatives[$outcomes[$i]]++;
            }
        }

        $score = 0.0;

        foreach ($truePositives as $label => $tp) {
            $fp = $falsePositives[$label];
            $fn = $falseNegatives[$label];

            $score += ((2 * $tp) / ((2 * $tp) + $fp + $fn) + self::EPSILON);
        }

        return $score / count($labels);
    }

    /**
     * Should this metric be minimized?
     *
     * @return bool
     */
    public function minimize() : bool
    {
        return false;
    }
}
