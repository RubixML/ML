<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;

use const Rubix\ML\EPSILON;

/**
 * V Measure
 *
 * V Measure is an entropy-based clustering metric that balances homogeneity and
 * completeness. It has the additional property of being symmetric in that the
 * predictions and ground-truth can be swapped without changing the score.
 *
 * References:
 * [1] A. Rosenberg et al. (2007). V-Measure: A conditional entropy-based
 * external cluster evaluation measure.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class VMeasure implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [0.0, 1.0];
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @return \Rubix\ML\EstimatorType[]
     */
    public function compatibility() : array
    {
        return [
            EstimatorType::clusterer(),
        ];
    }

    /**
     * Score a set of predictions.
     *
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        if (empty($predictions)) {
            return 0.0;
        }

        $homogeneity = (new Homogeneity())->score($predictions, $labels);
        $completeness = (new Completeness())->score($predictions, $labels);

        return 2.0 * ($homogeneity * $completeness)
            / (($homogeneity + $completeness) ?: EPSILON);
    }
}
