<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use InvalidArgumentException;

use const Rubix\ML\EPSILON;

/**
 * MAPE
 *
 * The *Mean Absolute Percentage Error* expresses the relative error of a set
 * of predictions and their labels as a percentage. It can be thought of as a
 * weighted version of Mean Absolute Error.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MAPE implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-INF, 0.];
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            Estimator::REGRESSOR,
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

        $error = 0.;

        foreach ($predictions as $i => $prediction) {
            $error += abs(($labels[$i] - $prediction)
                / ($prediction ?: EPSILON)) * 100.;
        }

        return -($error / count($predictions));
    }
}
