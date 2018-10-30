<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
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
     * Calculate the coefficient of determination i.e. R^2 from the predictions.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(Estimator $estimator, Dataset $testing) : float
    {
        if ($estimator->type() !== Estimator::REGRESSOR) {
            throw new InvalidArgumentException('This metric only works with'
                . ' regresors.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This metric requires a labeled'
                . ' testing set.');
        }

        if ($testing->numRows() < 0) {
            return 0.;
        }

        $mean = Stats::mean($testing->labels());

        $ssr = $sst = 0.;

        foreach ($estimator->predict($testing) as $i => $prediction) {
            $ssr += ($testing->label($i) - $prediction) ** 2;
            $sst += ($testing->label($i) - $mean) ** 2;
        }

        return 1. - ($ssr / ($sst ?: self::EPSILON));
    }
}
