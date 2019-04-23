<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use InvalidArgumentException;

use const Rubix\ML\EPSILON;

/**
 * F1 Score
 *
 * A weighted average of precision and recall with equal relative contribution.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class F1Score implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [0., 1.];
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

        $truePositives = $falsePositives = $falseNegatives =
            array_fill_keys($classes, 0);

        foreach ($predictions as $i => $prediction) {
            $label = $labels[$i];

            if ($prediction === $label) {
                $truePositives[$prediction]++;
            } else {
                $falsePositives[$prediction]++;
                $falseNegatives[$label]++;
            }
        }

        return array_sum(array_map(
            [$this, 'compute'],
            $truePositives,
            $falsePositives,
            $falseNegatives
        )) / count($classes);
    }

    /**
     * Compute the class f1 score.
     *
     * @param int $tp
     * @param int $fp
     * @param int $fn
     * @return float
     */
    public function compute(int $tp, int $fp, int $fn) : float
    {
        return (2 * $tp) / ((2 * $tp + $fp + $fn) ?: EPSILON);
    }
}
