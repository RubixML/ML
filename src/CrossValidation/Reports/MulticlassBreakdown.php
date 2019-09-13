<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use InvalidArgumentException;

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
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            Estimator::CLASSIFIER,
            Estimator::ANOMALY_DETECTOR,
        ];
    }

    /**
     * Generate the report.
     *
     * @param array $predictions
     * @param array $labels
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(array $predictions, array $labels) : array
    {
        $n = count($predictions);

        if ($n !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        $classes = array_unique(array_merge($predictions, $labels));

        $k = count($classes);

        $truePos = $trueNeg = $falsePos = $falseNeg = array_fill_keys($classes, 0);

        foreach ($predictions as $i => $prediction) {
            $label = $labels[$i];

            if ($prediction === $label) {
                $truePos[$prediction]++;
                
                foreach ($classes as $class) {
                    if ($class !== $prediction) {
                        $trueNeg[$class]++;
                    }
                }
            } else {
                $falsePos[$prediction]++;
                $falseNeg[$label]++;
            }
        }

        $averages = array_fill_keys([
            'accuracy', 'precision', 'recall', 'specificity', 'negative_predictive_value',
            'false_discovery_rate', 'miss_rate', 'fall_out', 'false_omission_rate',
            'f1_score', 'mcc', 'informedness', 'markedness',
        ], 0.);

        $counts = array_fill_keys([
            'true_positives', 'true_negatives', 'false_positives', 'false_negatives',
        ], 0);

        $overall = array_replace($averages, $counts);

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

            $f1score = 2. * (($precision * $recall))
                / (($precision + $recall) ?: EPSILON);

            $mcc = ($tp * $tn - $fp * $fn)
                / sqrt((($tp + $fp) * ($tp + $fn)
                * ($tn + $fp) * ($tn + $fn)) ?: EPSILON);

            $cardinality = $tp + $fn;

            $metrics = [];

            $metrics['accuracy'] = $accuracy;
            $metrics['precision'] = $precision;
            $metrics['recall'] = $recall;
            $metrics['specificity'] = $specificity;
            $metrics['negative_predictive_value'] = $npv;
            $metrics['false_discovery_rate'] = 1. - $precision;
            $metrics['miss_rate'] = 1. - $recall;
            $metrics['fall_out'] = 1. - $specificity;
            $metrics['false_omission_rate'] = 1. - $npv;
            $metrics['f1_score'] = $f1score;
            $metrics['informedness'] = $recall + $specificity - 1.;
            $metrics['markedness'] = $precision + $npv - 1.;
            $metrics['mcc'] = $mcc;
            $metrics['true_positives'] = $tp;
            $metrics['true_negatives'] = $tn;
            $metrics['false_positives'] = $fp;
            $metrics['false_negatives'] = $fn;
            $metrics['cardinality'] = $cardinality;
            $metrics['density'] = $cardinality / $n;

            $table[$label] = $metrics;

            $overall['accuracy'] += $accuracy;
            $overall['precision'] += $precision;
            $overall['recall'] += $recall;
            $overall['specificity'] += $specificity;
            $overall['negative_predictive_value'] += $npv;
            $overall['false_discovery_rate'] += 1. - $precision;
            $overall['miss_rate'] += 1. - $recall;
            $overall['fall_out'] += 1. - $specificity;
            $overall['false_omission_rate'] += 1. - $npv;
            $overall['f1_score'] += $f1score;
            $overall['informedness'] += $recall + $specificity - 1.;
            $overall['markedness'] += $precision + $npv - 1.;
            $overall['mcc'] += $mcc;
            $overall['true_positives'] += $tp;
            $overall['true_negatives'] += $tn;
            $overall['false_positives'] += $fp;
            $overall['false_negatives'] += $fn;
        }

        foreach (array_keys($averages) as $metric) {
            $overall[$metric] /= $k;
        }

        $overall['cardinality'] = $n;

        return [
            'overall' => $overall,
            'classes' => $table,
        ];
    }
}
