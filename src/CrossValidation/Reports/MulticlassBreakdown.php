<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
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
     * The classes to break down.
     *
     * @var array|null
     */
    protected $classes;

    /**
     * @param array|null $classes
     * @throws \InvalidArgumentException
     */
    public function __construct(?array $classes = null)
    {
        if (is_array($classes)) {
            $classes = array_unique($classes);

            foreach ($classes as $class) {
                if (!is_string($class) and !is_int($class)) {
                    throw new InvalidArgumentException('Class type must be a'
                        . ' string or integer, ' . gettype($class) . ' found.');
                }
            }
        }

        $this->classes = $classes;
    }

    /**
     * The estimator types that this report is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            Estimator::CLASSIFIER,
            Estimator::DETECTOR,
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
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        $classes = $this->classes ?: array_unique(array_merge($predictions, $labels));

        $n = count($predictions);

        $truePositives = $trueNegatives = $falsePositives = $falseNegatives =
            array_fill_keys($classes, 0);

        foreach ($predictions as $i => $prediction) {
            if (isset($truePositives[$prediction])) {
                $label = $labels[$i];

                if ($prediction === $label) {
                    $truePositives[$prediction]++;

                    foreach ($classes as $class) {
                        if ($class !== $prediction) {
                            $trueNegatives[$class]++;
                        }
                    }
                } else {
                    $falsePositives[$prediction]++;
                    $falseNegatives[$label]++;
                }
            }
        }

        $k = count($classes);

        $overall = array_fill_keys([
            'accuracy', 'precision', 'recall', 'specificity', 'negative_predictive_value',
            'false_discovery_rate', 'miss_rate', 'fall_out', 'false_omission_rate',
            'f1_score', 'mcc', 'informedness', 'markedness', 'true_positives',
            'true_negatives', 'false_positives', 'false_negatives',
        ], 0);

        $table = array_fill_keys($classes, []);

        foreach ($truePositives as $label => $tp) {
            $tn = $trueNegatives[$label];
            $fp = $falsePositives[$label];
            $fn = $falseNegatives[$label];

            $accuracy = ($tp + $tn) / ($tp + $tn + $fp + $fn);
            $precision = $tp / (($tp + $fp) ?: self::EPSILON);
            $recall = $tp / (($tp + $fn) ?: self::EPSILON);
            $specificity = $tn / (($tn + $fp) ?: self::EPSILON);
            $npv = $tn / (($tn + $fn) ?: self::EPSILON);
            $cardinality = $tp + $fn;

            $f1 = 2. * (($precision * $recall))
                / (($precision + $recall) ?: self::EPSILON);

            $mcc = ($tp * $tn - $fp * $fn)
                / sqrt((($tp + $fp) * ($tp + $fn)
                * ($tn + $fp) * ($tn + $fn)) ?: self::EPSILON);

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
            $metrics['f1_score'] = $f1;
            $metrics['mcc'] = $mcc;
            $metrics['informedness'] = $recall + $specificity - 1.;
            $metrics['markedness'] = $precision + $npv - 1.;
            $metrics['true_positives'] = $tp;
            $metrics['true_negatives'] = $tn;
            $metrics['false_positives'] = $fp;
            $metrics['false_negatives'] = $fn;
            $metrics['cardinality'] = $cardinality;
            $metrics['density'] = $cardinality / $n;

            $table[$label] = $metrics;

            $overall['accuracy'] += $accuracy / $k;
            $overall['precision'] += $precision / $k;
            $overall['recall'] += $recall / $k;
            $overall['specificity'] += $specificity / $k;
            $overall['negative_predictive_value'] += $npv / $k;
            $overall['false_discovery_rate'] += (1. - $precision) / $k;
            $overall['miss_rate'] += (1. - $recall) / $k;
            $overall['fall_out'] += (1. - $specificity) / $k;
            $overall['false_omission_rate'] += (1. - $npv) / $k;
            $overall['f1_score'] += $f1 / $k;
            $overall['mcc'] += $mcc / $k;
            $overall['informedness'] += ($recall + $specificity - 1.) / $k;
            $overall['markedness'] += ($precision + $npv - 1.) / $k;
            $overall['true_positives'] += $tp;
            $overall['true_negatives'] += $tn;
            $overall['false_positives'] += $fp;
            $overall['false_negatives'] += $fn;
        }

        $overall['cardinality'] = $n;
        $overall['density'] = 1.;

        return [
            'overall' => $overall,
            'label' => $table,
        ];
    }
}
