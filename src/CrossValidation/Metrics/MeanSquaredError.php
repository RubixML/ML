<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use InvalidArgumentException;

/**
 * Mean Squared Error
 *
 * A metric that punishes bad predictions the worse they get by taking the
 * squared error and averaging it over the testing set.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MeanSquaredError implements Metric
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
     * @param  array  $predictions
     * @param  array  $labels
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
            $error += ($labels[$i] - $prediction) ** 2;
        }

        return -($error / count($predictions));
    }
}
