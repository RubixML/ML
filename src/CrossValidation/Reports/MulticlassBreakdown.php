<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use InvalidArgumentException;

use function count;

use const Rubix\ML\EPSILON;

/**
 * Multiclass Breakdown
 *
 * A classification and anomaly detection report that drills down into unique class
 * statistics as well as provide an overall picture. The report includes metrics
 * such as Accuracy, F1 Score, MCC, Precision, Recall, Fall Out, and Miss Rate.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MulticlassBreakdown implements Report
{
    /**
     * The estimator types that this report is compatible with.
     *
     * @return \Rubix\ML\EstimatorType[]
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
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @throws \InvalidArgumentException
     * @return mixed[]
     */
    public function generate(array $predictions, array $labels) : array
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('Number of predictions'
                . ' and labels must be equal.');
        }

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
            'accuracy', 'accuracy_balanced', 'f1_score', 'precision', 'recall', 'specificity',
            'negative_predictive_value', 'false_discovery_rate', 'miss_rate', 'fall_out',
            'false_omission_rate', 'threat_score', 'mcc', 'informedness', 'markedness',
        ], 0.0);

        $counts = array_fill_keys([
            'true_positives', 'true_negatives', 'false_positives', 'false_negatives',
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

            $threatScore = $tp / ($tp + $fn + $fp);

            $f1score = 2.0 * (($precision * $recall))
                / (($precision + $recall) ?: EPSILON);

            $mcc = ($tp * $tn - $fp * $fn)
                / sqrt((($tp + $fp) * ($tp + $fn)
                * ($tn + $fp) * ($tn + $fn)) ?: EPSILON);

            $cardinality = $tp + $fn;

            $table[$label] = [
                'accuracy' => $accuracy,
                'accuracy_balanced' => ($recall + $specificity) / 2.0,
                'f1_score' => $f1score,
                'precision' => $precision,
                'recall' => $recall,
                'specificity' => $specificity,
                'negative_predictive_value' => $npv,
                'false_discovery_rate' => 1.0 - $precision,
                'miss_rate' => 1.0 - $recall,
                'fall_out' => 1.0 - $specificity,
                'false_omission_rate' => 1.0 - $npv,
                'threat_score' => $threatScore,
                'informedness' => $recall + $specificity - 1.0,
                'markedness' => $recall + $specificity - 1.0,
                'mcc' => $mcc,
                'true_positives' => $tp,
                'true_negatives' => $tn,
                'false_positives' => $fp,
                'false_negatives' => $fn,
                'cardinality' => $cardinality,
                'percentage' => $cardinality / $n * 100.00,
            ];

            $overall['accuracy'] += $accuracy;
            $overall['accuracy_balanced'] += ($recall + $specificity) / 2.0;
            $overall['f1_score'] += $f1score;
            $overall['precision'] += $precision;
            $overall['recall'] += $recall;
            $overall['specificity'] += $specificity;
            $overall['negative_predictive_value'] += $npv;
            $overall['false_discovery_rate'] += 1.0 - $precision;
            $overall['miss_rate'] += 1.0 - $recall;
            $overall['fall_out'] += 1.0 - $specificity;
            $overall['false_omission_rate'] += 1.0 - $npv;
            $overall['threat_score'] += $threatScore;
            $overall['informedness'] += $recall + $specificity - 1.0;
            $overall['markedness'] += $precision + $npv - 1.0;
            $overall['mcc'] += $mcc;
            $overall['true_positives'] += $tp;
            $overall['true_negatives'] += $tn;
            $overall['false_positives'] += $fp;
            $overall['false_negatives'] += $fn;
        }

        foreach (array_keys($averages) as $metric) {
            $overall[$metric] /= $k;
        }

        $overall += [
            'cardinality' => $n,
        ];

        return [
            'overall' => $overall,
            'classes' => $table,
        ];
    }
}
