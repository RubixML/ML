<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;
use Stringable;

use const Rubix\ML\EPSILON;

/**
 * MCC
 *
 * Matthews Correlation Coefficient (MCC) measures the quality of a classification by taking
 * into account true and false positives and negatives. It is generally regarded as a
 * balanced measure which can be used even if the classes are of very different sizes. The
 * MCC is a correlation coefficient between the observed and predicted binary classifications.
 * A coefficient of 1 represents a perfect prediction, 0 no better than random prediction, and
 * −1 indicates total disagreement between prediction and observation.
 *
 * References:
 * [1] B. W. Matthews. (1975). Decision of the Predicted and Observed Secondary
 * Structure of T4 Phage Lysozyme.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MCC implements Metric, Stringable
{
    /**
     * Compute the class mcc score.
     *
     * @param int $tp
     * @param int $tn
     * @param int $fp
     * @param int $fn
     * @return float
     */
    public static function compute(int $tp, int $tn, int $fp, int $fn) : float
    {
        return ($tp * $tn - $fp * $fn) / (sqrt(($tp + $fp) * ($tp + $fn) * ($tn + $fp) * ($tn + $fn)) ?: EPSILON);
    }

    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-1.0, 1.0];
    }

    /**
     * The estimator types that this metric is compatible with.
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
     * Score a set of predictions.
     *
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        PredictionAndLabelCountsAreEqual::with($predictions, $labels)->check();

        if (empty($predictions)) {
            return 0.0;
        }

        $classes = array_unique(array_merge($predictions, $labels));

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

        $scores = array_map([self::class, 'compute'], $truePos, $trueNeg, $falsePos, $falseNeg);

        return Stats::mean($scores);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'MCC';
    }
}
