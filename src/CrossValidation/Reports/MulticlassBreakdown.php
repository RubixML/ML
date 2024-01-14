<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Report;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;

use function count;
use function array_fill_keys;
use function array_merge;
use function array_unique;
use function array_keys;

use const Rubix\ML\EPSILON;

/**
 * Multiclass Breakdown
 *
 * A multiclass classification report that computes a number of metrics (Accuracy, Precision,
 * Recall, etc.) derived from their confusion matrix on an overall and individual class basis.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MulticlassBreakdown implements ReportGenerator
{
    /**
     * The estimator types that this report is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\EstimatorType>
     */
    public function compatibility() : array
    {
        return [
            EstimatorType::classifier(),
            EstimatorType::anomalyDetector(),
        ];
    }

    /**
     * Generate the report.
     *
     * @param list<string|int> $predictions
     * @param list<string|int> $labels
     * @return Report
     */
    public function generate(array $predictions, array $labels) : Report
    {
        PredictionAndLabelCountsAreEqual::with($predictions, $labels)->check();

        $classes = array_unique(array_merge($predictions, $labels));

        $n = count($predictions);
        $k = count($classes);

        $truePos = $trueNeg = $falsePos = $falseNeg = array_fill_keys($classes, 0);

        foreach ($predictions as $i => $prediction) {
            $label = $labels[$i];

            if ($prediction == $label) {
                ++$truePos[$prediction];

                foreach ($classes as $class) {
                    if ($class != $prediction) {
                        ++$trueNeg[$class];
                    }
                }
            } else {
                ++$falsePos[$prediction];
                ++$falseNeg[$label];
            }
        }

        $averages = array_fill_keys([
            'accuracy', 'balanced accuracy', 'f1 score', 'precision', 'recall', 'specificity',
            'negative predictive value', 'false discovery rate', 'miss rate', 'fall out',
            'false omission rate', 'mcc', 'informedness', 'markedness',
        ], 0.0);

        $counts = array_fill_keys([
            'true positives', 'true negatives', 'false positives', 'false negatives',
        ], 0);

        $overall = $averages + $counts;

        $table = [];

        foreach ($truePos as $label => $tp) {
            $tn = $trueNeg[$label];
            $fp = $falsePos[$label];
            $fn = $falseNeg[$label];

            $accuracy = ($tp + $tn) / (($tp + $tn + $fp + $fn) ?: EPSILON);
            $precision = $tp / (($tp + $fp) ?: EPSILON);
            $recall = $tp / (($tp + $fn) ?: EPSILON);
            $specificity = $tn / (($tn + $fp) ?: EPSILON);
            $npv = $tn / (($tn + $fn) ?: EPSILON);

            $f1score = 2.0 * (($precision * $recall)
                / (($precision + $recall) ?: EPSILON));

            $mcc = ($tp * $tn - $fp * $fn)
                / (sqrt(($tp + $fp) * ($tp + $fn)
                * ($tn + $fp) * ($tn + $fn)) ?: EPSILON);

            $cardinality = $tp + $fn;

            $table[$label] = [
                'accuracy' => $accuracy,
                'balanced accuracy' => ($recall + $specificity) / 2.0,
                'f1 score' => $f1score,
                'precision' => $precision,
                'recall' => $recall,
                'specificity' => $specificity,
                'negative predictive value' => $npv,
                'false discovery rate' => 1.0 - $precision,
                'miss rate' => 1.0 - $recall,
                'fall out' => 1.0 - $specificity,
                'false omission rate' => 1.0 - $npv,
                'informedness' => $recall + $specificity - 1.0,
                'markedness' => $precision + $npv - 1.0,
                'mcc' => $mcc,
                'true positives' => $tp,
                'true negatives' => $tn,
                'false positives' => $fp,
                'false negatives' => $fn,
                'cardinality' => $cardinality,
                'proportion' => $cardinality / $n,
            ];

            $overall['accuracy'] += $accuracy;
            $overall['balanced accuracy'] += ($recall + $specificity) / 2.0;
            $overall['f1 score'] += $f1score;
            $overall['precision'] += $precision;
            $overall['recall'] += $recall;
            $overall['specificity'] += $specificity;
            $overall['negative predictive value'] += $npv;
            $overall['false discovery rate'] += 1.0 - $precision;
            $overall['miss rate'] += 1.0 - $recall;
            $overall['fall out'] += 1.0 - $specificity;
            $overall['false omission rate'] += 1.0 - $npv;
            $overall['informedness'] += $recall + $specificity - 1.0;
            $overall['markedness'] += $precision + $npv - 1.0;
            $overall['mcc'] += $mcc;
            $overall['true positives'] += $tp;
            $overall['true negatives'] += $tn;
            $overall['false positives'] += $fp;
            $overall['false negatives'] += $fn;
        }

        foreach (array_keys($averages) as $metric) {
            $overall[$metric] /= $k;
        }

        $overall += [
            'cardinality' => $n,
        ];

        return new Report([
            'overall' => $overall,
            'classes' => $table,
        ]);
    }
}
