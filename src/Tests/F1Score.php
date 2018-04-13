<?php

namespace Rubix\Engine\Tests;

use MathPHP\Statistics\Average;
use InvalidArgumentException;

class F1Score extends Test
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

        $f1scores = $truePositives = $falsePositives = $falseNegatives = [];

        foreach (array_unique(array_merge($predictions, $outcomes)) as $label) {
            $f1scores[$label] = $truePositives[$label] = $falsePositives[$label] = $falseNegatives[$label] = 0;
        }

        foreach ($predictions as $i => $prediction) {
            if ($prediction === $outcomes[$i]) {
                $truePositives[$prediction]++;
            } else {
                $falsePositives[$prediction]++;
                $falseNegatives[$outcomes[$i]]++;
            }
        }

        foreach ($truePositives as $label => $count) {
            $precision = $count / ($count + $falsePositives[$label] + self::EPSILON);
            $recall = $count / ($count + $falseNegatives[$label] + self::EPSILON);

            $f1scores[$label] = 2.0 * (($precision * $recall) / ($precision + $recall + self::EPSILON));
        }

        $average = Average::mean($f1scores);
        $best = max($f1scores);
        $worst = min($f1scores);

        $this->logger->log('Average F1 score: ' . number_format($average, 3));
        $this->logger->log('Best F1 score: ' . number_format($best, 3) . ' (label: ' . (string) array_search($best, $f1scores) . ')');
        $this->logger->log('Worst F1 score: ' . number_format($worst, 3) . ' (label: ' . (string) array_search($worst, $f1scores) . ')');

        return $average;
    }
}
