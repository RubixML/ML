<?php

namespace Rubix\Engine\Metrics\Reports;

use Rubix\Engine\Metrics\F1Score;
use Rubix\Engine\Metrics\Accuracy;
use InvalidArgumentException;

class ClassificationReport implements Report
{
    /**
     * Prepare the classification report. This involves calculating a number of
     * useful metrics on a per outcome basis.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @throws \InvalidArgumentException
     * @return void
     */
    public function generate(array $predictions, array $outcomes) : array
    {
        if (count($predictions) !== count($outcomes)) {
            throw new InvalidArgumentException('The number of outcomes must match the number of predictions.');
        }

        $table = $truePositives = $trueNegatives = $falsePositives = $falseNegatives = [];

        foreach (array_unique(array_merge($predictions, $outcomes)) as $label) {
            $truePositives[$label] = $trueNegatives[$label] = $falsePositives[$label] = $falseNegatives[$label] = 0;
            $table[$label] = [];
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

        foreach ($truePositives as $label => $tp) {
            $tn = $trueNegatives[$label];
            $fp = $falsePositives[$label];
            $fn = $falseNegatives[$label];

            $table[$label]['accuracy'] = ($tp + $tn) / ($tp + $tn + $fp + $fn + self::EPSILON);
            $table[$label]['precision'] = $tp / ($tp + $fp + self::EPSILON);
            $table[$label]['recall'] = $tp / ($tp + $fn + self::EPSILON);
            $table[$label]['specificity'] = $tn / ($tn + $fp + self::EPSILON);
            $table[$label]['miss_rate'] = 1 - $table[$label]['recall'];
            $table[$label]['fall_out'] = 1 - $table[$label]['specificity'];
            $table[$label]['f1_score'] = 2.0 * (($table[$label]['precision'] * $table[$label]['recall']) / ($table[$label]['precision'] + $table[$label]['recall'] + self::EPSILON));
            $table[$label]['mcc'] = ($tp * $tn - $fp * $fn) / (sqrt(($tp + $fp) * ($tp + $fn) * ($tn + $fp) * ($tn + $fn)) + self::EPSILON);
            $table[$label]['informedness'] = $table[$label]['recall'] + $table[$label]['specificity'] - 1;
        }

        return [
            'average' => [
                'accuracy' => (new Accuracy())->score($predictions, $outcomes),
                'f1_score' => (new F1Score())->score($predictions, $outcomes),
            ],
            'labels' => $table,
        ];
    }
}
