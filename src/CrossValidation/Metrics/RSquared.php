<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;

/**
 * R Squared
 *
 * The *coefficient of determination* or R Squared is the proportion of the
 * variance in the dependent variable that is predictable from the independent
 * variable(s).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RSquared implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-INF, 1.];
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

        $mean = Stats::mean($labels);

        $ssr = $sst = 0.;

        foreach ($predictions as $i => $prediction) {
            $label = $labels[$i];

            $ssr += ($label - $prediction) ** 2;
            $sst += ($label - $mean) ** 2;
        }

        return 1. - ($ssr / ($sst ?: self::EPSILON));
    }
}
