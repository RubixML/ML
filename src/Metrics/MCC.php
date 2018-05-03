<?php

namespace Rubix\Engine\Metrics;

use InvalidArgumentException;

class MCC implements Classification
{
    /**
     * Test the Matthews correlation coefficient of the predictions. Outputs a
     * number between -1 and 1.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @return float
     */
    public function score(array $predictions, array $outcomes) : float
    {
        if (count($predictions) !== count($outcomes)) {
            throw new InvalidArgumentException('The number of outcomes must match the number of predictions.');
        }

        $labels = array_unique(array_merge($predictions, $outcomes));

        $truePositives = $trueNegatives = $falsePositives = $falseNegatives = [];

        foreach ($labels as $label) {
            $truePositives[$label] = $trueNegatives[$label] = $falsePositives[$label] = $falseNegatives[$label] = 0;
        }

        foreach ($predictions as $i => $prediction) {
            if ($prediction === $outcomes[$i]) {
                $truePositives[$prediction]++;

                foreach ($labels as $label) {
                    if ($label !== $prediction) {
                        $trueNegatives[$label]++;
                    }
                }
            } else {
                $falsePositives[$prediction]++;
                $falseNegatives[$outcomes[$i]]++;
            }
        }

        $score = 0.0;

        foreach ($truePositives as $label => $tp) {
            $tn = $trueNegatives[$label];
            $fp = $falsePositives[$label];
            $fn = $falseNegatives[$label];

            $score += (($tp * $tn - $fp * $fn) / (sqrt(($tp + $fp) * ($tp + $fn) * ($tn + $fp) * ($tn + $fn)) + self::EPSILON));
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
