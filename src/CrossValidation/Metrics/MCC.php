<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;

use const Rubix\ML\EPSILON;

/**
 * MCC
 *
 * Matthews Correlation Coefficient measures the quality of a classification. It
 * takes into account true and false positives and negatives and is generally
 * regarded as a balanced measure which can be used even if the classes are of
 * very different sizes. The MCC is in essence a correlation coefficient between
 * the observed and predicted binary classifications; it returns a value between
 * −1 and +1. A coefficient of +1 represents a perfect prediction, 0 no better
 * than random prediction and −1 indicates total disagreement between prediction
 * and observation.
 *
 * References:
 * [1] B. W. Matthews. (1975). Decision of the Predicted and Observed Secondary
 * Structure of T4 Phage Lysozyme.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MCC implements Metric
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
    public static function mcc(int $tp, int $tn, int $fp, int $fn) : float
    {
        return ($tp * $tn - $fp * $fn)
            / (sqrt(($tp + $fp) * ($tp + $fn)
            * ($tn + $fp) * ($tn + $fn)) ?: EPSILON);
    }

    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-1., 1.];
    }

    /**
     * The estimator types that this metric is compatible with.
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
     * Score a set of predictions.
     *
     * @param array $predictions
     * @param array $labels
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        if (empty($predictions)) {
            return 0.;
        }

        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        $classes = array_unique(array_merge($predictions, $labels));

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

        return Stats::mean(
            array_map([$this, 'mcc'], $truePos, $trueNeg, $falsePos, $falseNeg)
        );
    }
}
