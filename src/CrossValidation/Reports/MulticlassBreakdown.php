<?php

namespace Rubix\ML\CrossValidation\Reports;

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
     * @param  array|null  $classes
     * @throws \InvalidArgumentException
     * @return void
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
     * Generate the report.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(array $predictions, array $labels) : array
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        if (is_null($this->classes)) {
            $classes = array_unique(array_merge($predictions, $labels));
        } else {
            $classes = $this->classes;
        }

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

            $table[$label]['accuracy'] = $accuracy;
            $table[$label]['precision'] = $precision;
            $table[$label]['recall'] = $recall;
            $table[$label]['specificity'] = $specificity;
            $table[$label]['negative_predictive_value'] = $npv;
            $table[$label]['false_discovery_rate'] = 1. - $precision;
            $table[$label]['miss_rate'] = 1. - $recall;
            $table[$label]['fall_out'] = 1. - $specificity;
            $table[$label]['false_omission_rate'] = 1. - $npv;
            $table[$label]['f1_score'] = $f1;
            $table[$label]['mcc'] = $mcc;
            $table[$label]['informedness'] = $recall + $specificity - 1.;
            $table[$label]['markedness'] = $precision + $npv - 1.;
            $table[$label]['true_positives'] = $tp;
            $table[$label]['true_negatives'] = $tn;
            $table[$label]['false_positives'] = $fp;
            $table[$label]['false_negatives'] = $fn;
            $table[$label]['cardinality'] = $cardinality;
            $table[$label]['density'] = $cardinality / $n;
        }

        $overall = array_fill_keys([
            'accuracy', 'precision', 'recall', 'specificity', 'negative_predictive_value',
            'false_discovery_rate', 'miss_rate', 'fall_out', 'false_omission_rate',
            'f1_score', 'mcc', 'informedness', 'markedness',
        ], 0.);

        $k = count($classes);

        foreach ($table as $metrics) {
            foreach ($overall as $metric => &$value) {
                $value += $metrics[$metric] / $k;
            }
        }

        return [
            'overall' => $overall,
            'label' => $table,
        ];
    }
}
