<?php

namespace Rubix\ML\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\Classifier;
use InvalidArgumentException;

/**
 * Multiclass Breakdown
 *
 * A Report that drills down in to each unique class outcome. The report
 * includes metrics such as Accuracy, F1 Score, MCC, Precision, Recall,
 * Cardinality, Miss Rate, and more.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MulticlassBreakdown implements Report
{
    /**
     * Prepare the classification report. This involves calculating a number of
     * useful metrics on a per outcome basis.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(Estimator $estimator, Dataset $testing) : array
    {
        if (!$estimator instanceof Classifier) {
            throw new InvalidArgumentException('This report only works on'
                . ' classifiers.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This report requires a'
                . ' Labeled testing set.');
        }

        if ($testing->numRows() === 0) {
            throw new InvalidArgumentException('Testing set must contain at'
                . ' least one sample.');
        }

        $predictions = $estimator->predict($testing);

        $labels = $testing->labels();

        $classes = array_unique(array_merge($predictions, $labels));

        $truePositives = $trueNegatives = $falsePositives = $falseNegatives =
            array_fill_keys($classes, 0);

        foreach ($predictions as $i => $outcome) {
            if ($outcome === $labels[$i]) {
                $truePositives[$outcome]++;

                foreach ($classes as $class) {
                    if ($class !== $outcome) {
                        $trueNegatives[$class]++;
                    }
                }
            } else {
                $falsePositives[$outcome]++;
                $falseNegatives[$labels[$i]]++;
            }
        }

        $table = array_fill_keys($classes, []);

        $overall = array_fill_keys([
            'accuracy', 'precision', 'recall', 'specificity', 'miss_rate',
            'fall_out', 'f1_score', 'mcc', 'informedness',
        ], 0.0);

        foreach ($truePositives as $label => $tp) {
            $tn = $trueNegatives[$label];
            $fp = $falsePositives[$label];
            $fn = $falseNegatives[$label];

            $table[$label]['accuracy'] = ($tp + $tn) / ($tp + $tn + $fp + $fn);
            $table[$label]['precision'] = ($tp + self::EPSILON) / ($tp + $fp + self::EPSILON);
            $table[$label]['recall'] = ($tp + self::EPSILON) / ($tp + $fn + self::EPSILON);
            $table[$label]['specificity'] = ($tn + self::EPSILON) / ($tn + $fp + self::EPSILON);
            $table[$label]['miss_rate'] = 1 - $table[$label]['recall'];
            $table[$label]['fall_out'] = 1 - $table[$label]['specificity'];
            $table[$label]['f1_score'] = 2.0 * (($table[$label]['precision']
                * $table[$label]['recall']) + self::EPSILON)
                / ($table[$label]['precision'] + $table[$label]['recall'] + self::EPSILON);
            $table[$label]['informedness'] = $table[$label]['recall']
                + $table[$label]['specificity'] - 1;
            $table[$label]['mcc'] = (($tp * $tn - $fp * $fn) + self::EPSILON)
                / (sqrt(($tp + $fp) * ($tp + $fn) * ($tn + $fp) * ($tn + $fn))
                + self::EPSILON);
            $table[$label]['true_positives'] = $tp;
            $table[$label]['true_negatives'] = $tn;
            $table[$label]['false_positives'] = $fp;
            $table[$label]['false_negatives'] = $fn;
            $table[$label]['cardinality'] = $tp + $fn;
            $table[$label]['density'] = $table[$label]['cardinality']
                / count($predictions);

            $overall['accuracy'] += $table[$label]['accuracy'];
            $overall['precision'] += $table[$label]['precision'];
            $overall['recall'] += $table[$label]['recall'];
            $overall['specificity'] += $table[$label]['specificity'];
            $overall['miss_rate'] += $table[$label]['miss_rate'];
            $overall['fall_out'] += $table[$label]['fall_out'];
            $overall['f1_score'] += $table[$label]['f1_score'];
            $overall['mcc'] += $table[$label]['mcc'];
            $overall['informedness'] += $table[$label]['informedness'];
        }

        $n = count($classes);

        foreach ($overall as $metric => &$score) {
            $score /= $n;
        }

        return [
            'overall' => $overall,
            'label' => $table,
        ];
    }
}
