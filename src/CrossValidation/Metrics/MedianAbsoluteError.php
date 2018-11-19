<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;

/**
 * Median Absolute Error
 *
 * Median Absolute Error (MAE) is a robust measure of the error that ignores
 * highly erroneous predictions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MedianAbsoluteError implements Metric
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

        $errors = [];

        foreach ($predictions as $i => $prediction) {
            $errors[] = abs($labels[$i] - $prediction);
        }

        return -Stats::median($errors);
    }
}
