<?php

namespace Rubix\ML\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
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
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(Estimator $estimator, Dataset $testing) : array
    {
        if ($estimator->type() !== Estimator::CLASSIFIER) {
            throw new InvalidArgumentException('This report only works with'
                . ' classifiers.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This report requires a'
                . ' Labeled testing set.');
        }

        $n = $testing->numRows();

        if ($n === 0) {
            throw new InvalidArgumentException('Testing set must contain at'
                . ' least one sample.');
        }

        $predictions = $estimator->predict($testing);

        $labels = $testing->labels();

        if (is_null($this->classes)) {
            $classes = array_unique(array_merge($predictions, $labels));
        } else {
            $classes = $this->classes;
        }

        $truePositives = $trueNegatives = $falsePositives = $falseNegatives =
            array_fill_keys($classes, 0);

        foreach ($predictions as $i => $outcome) {
            if (!isset($truePositives[$outcome])) {
                continue 1;
            }

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

        foreach ($truePositives as $label => $tp) {
            $tn = $trueNegatives[$label];
            $fp = $falsePositives[$label];
            $fn = $falseNegatives[$label];

            $accuracy = ($tp + $tn) / ($tp + $tn + $fp + $fn);
            $precision = $tp / (($tp + $fp) ?: self::EPSILON);
            $recall = $tp / (($tp + $fn) ?: self::EPSILON);
            $specificity = $tn / (($tn + $fp) ?: self::EPSILON);
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
            $table[$label]['miss_rate'] = 1. - $recall;
            $table[$label]['fall_out'] = 1. - $specificity;
            $table[$label]['f1_score'] = $f1;
            $table[$label]['informedness'] = $recall + $specificity - 1;
            $table[$label]['mcc'] = $mcc;
            $table[$label]['true_positives'] = $tp;
            $table[$label]['true_negatives'] = $tn;
            $table[$label]['false_positives'] = $fp;
            $table[$label]['false_negatives'] = $fn;
            $table[$label]['cardinality'] = $cardinality;
            $table[$label]['density'] = $cardinality / $n;
        }

        $overall = array_fill_keys([
            'accuracy', 'precision', 'recall', 'specificity', 'miss_rate',
            'fall_out', 'f1_score', 'informedness', 'mcc',
        ], 0.);

        $k = count($classes);

        foreach ($table as $row) {
            foreach ($overall as $metric => &$value) {
                $value += $row[$metric] / $k;
            }
        }

        return [
            'overall' => $overall,
            'label' => $table,
        ];
    }
}
