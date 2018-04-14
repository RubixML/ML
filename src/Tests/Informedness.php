<?php

namespace Rubix\Engine\Tests;

use MathPHP\Statistics\Average;
use InvalidArgumentException;

class Informedness extends Test
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

        $informedness = $truePositives = $trueNegatives = $falsePositives = $falseNegatives = [];

        foreach (array_unique(array_merge($predictions, $outcomes)) as $label) {
            $informedness[$label] = $truePositives[$label] = $trueNegatives[$label] = $falsePositives[$label] = $falseNegatives[$label] = 0;
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
            $recall = $count / ($count + $falseNegatives[$label] + self::EPSILON);
            $specificity = $trueNegatives[$label] / ($trueNegatives[$label] + $falsePositives[$label] + self::EPSILON);

            $informedness[$label] = $recall + $specificity - 1;
        }

        $average = Average::mean($informedness);
        $best = max($informedness);
        $worst = min($informedness);

        $this->logger->log('Average informedness: ' . number_format($average, 5));
        $this->logger->log('Best informedness: ' . number_format($best, 5) . ' (label: ' . (string) array_search($best, $informedness) . ')');
        $this->logger->log('Worst informedness: ' . number_format($worst, 5) . ' (label: ' . (string) array_search($worst, $informedness) . ')');

        return $average;
    }
}
