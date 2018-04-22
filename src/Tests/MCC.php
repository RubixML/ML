<?php

namespace Rubix\Engine\Tests;

use MathPHP\Statistics\Average;
use InvalidArgumentException;

class MCC extends Test
{
    /**
     * Test the Matthews correlation coefficient of the predictions.
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

        $mcc = $truePositives = $trueNegatives = $falsePositives = $falseNegatives = [];

        foreach (array_unique(array_merge($predictions, $outcomes)) as $label) {
            $mcc[$label] = $truePositives[$label] = $trueNegatives[$label] = $falsePositives[$label] = $falseNegatives[$label] = 0;
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

            $mcc[$label] = ($count * $tn - $fp * $fn) / sqrt(($count + $fp) * ($count + $fn) * ($tn + $fp) * ($tn + $fn)) + self::EPSILON;
        }

        $average = Average::mean($mcc);
        $best = max($mcc);
        $worst = min($mcc);

        $this->logger->log('Average MCC: ' . number_format($average, 5));
        $this->logger->log('Best MCC: ' . number_format($best, 5) . ' (label: ' . (string) array_search($best, $mcc) . ')');
        $this->logger->log('Worst MCC: ' . number_format($worst, 5) . ' (label: ' . (string) array_search($worst, $mcc) . ')');

        return $average;
    }
}
