<?php

namespace Rubix\Engine\Tests;

use MathPHP\Statistics\Average;
use InvalidArgumentException;

class Accuracy extends Test
{
    /**
     * Test the accuracy of the predictions.
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

        $accuracy = $truePositives = $trueNegatives = $falsePositives = $falseNegatives = [];

        foreach (array_unique(array_merge($predictions, $outcomes)) as $label) {
            $accuracy[$label] = $truePositives[$label] = $trueNegatives[$label] = $falsePositives[$label] = $falseNegatives[$label] = 0;
        }

        foreach ($predictions as $i => $prediction) {
            if ($prediction === $outcomes[$i]) {
                $truePositives[$prediction]++;
                $trueNegatives[$outcomes[$i]]++;
            } else {
                $falsePositives[$prediction]++;
                $falseNegatives[$outcomes[$i]]++;
            }
        }

        foreach ($truePositives as $label => $count) {
            $tn = $trueNegatives[$label];
            $fp = $falsePositives[$label];
            $fn = $falseNegatives[$label];

            $accuracy[$label] = ($count + $tn) / ($count + $tn + $fp + $fn);
        }

        $average = Average::mean($accuracy);
        $best = max($accuracy);
        $worst = min($accuracy);

        $this->logger->log('Average Accuracy: ' . number_format($average, 5));
        $this->logger->log('Best Accuracy: ' . number_format($best, 5) . ' (label: ' . (string) array_search($best, $accuracy) . ')');
        $this->logger->log('Worst Accuracy: ' . number_format($worst, 5) . ' (label: ' . (string) array_search($worst, $accuracy) . ')');

        return $average;
    }
}
