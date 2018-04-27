<?php

namespace Rubix\Engine\Reports;

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

        $labels = array_unique(array_merge($predictions, $outcomes));
        $table = [];

        $truePositives = $trueNegatives = $falsePositives = $falseNegatives = [];

        foreach ($labels as $label) {
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

        $overall = array_fill_keys([
            'accuracy', 'precision', 'recall', 'specificity', 'miss_rate', 'fall_out', 'f1_score', 'informedness',
        ], 0.0);

        foreach ($truePositives as $label => $tp) {
            $tn = $trueNegatives[$label];
            $fp = $falsePositives[$label];
            $fn = $falseNegatives[$label];

            $table[$label]['accuracy'] = ($tp + $tn) / ($tp + $tn + $fp + $fn);
            $table[$label]['precision'] = $tp / ($tp + $fp + self::EPSILON);
            $table[$label]['recall'] = $tp / ($tp + $fn + self::EPSILON);
            $table[$label]['specificity'] = $tn / ($tn + $fp + self::EPSILON);
            $table[$label]['miss_rate'] = 1 - $table[$label]['recall'];
            $table[$label]['fall_out'] = 1 - $table[$label]['specificity'];
            $table[$label]['f1_score'] = 2.0 * (($table[$label]['precision'] * $table[$label]['recall']) / ($table[$label]['precision'] + $table[$label]['recall'] + self::EPSILON));
            $table[$label]['informedness'] = $table[$label]['recall'] + $table[$label]['specificity'] - 1;
            $table[$label]['cardinality'] = $tp + $fn;
            $table[$label]['density'] = $table[$label]['cardinality'] / count($outcomes);

            $overall['accuracy'] += $table[$label]['accuracy'];
            $overall['precision'] += $table[$label]['precision'];
            $overall['recall'] += $table[$label]['recall'];
            $overall['specificity'] += $table[$label]['specificity'];
            $overall['miss_rate'] += $table[$label]['miss_rate'];
            $overall['fall_out'] += $table[$label]['fall_out'];
            $overall['f1_score'] += $table[$label]['f1_score'];
            $overall['informedness'] += $table[$label]['informedness'];
        }

        $n = count($labels);

        return [
            'overall' => array_map(function ($metric) use ($n) {
                return $metric / $n;
            }, $overall),
            'label' => $table,
        ];
    }
}
