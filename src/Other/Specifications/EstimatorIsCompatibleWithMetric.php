<?php

namespace Rubix\ML\Other\Specifications;

use Rubix\ML\Estimator;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\CrossValidation\Metrics\Metric;
use InvalidArgumentException;

class EstimatorIsCompatibleWithMetric
{
    /**
     * Perform a check.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\CrossValidation\Metrics\Metric  $metric
     * @throws \InvalidArgumentException
     * @return void
     */
    public static function check(Estimator $estimator, Metric $metric) : void
    {
        if (!in_array($estimator->type(), $metric->compatibility())) {
            throw new InvalidArgumentException(Params::shortName($metric)
                . ' is not compatible with '
                . Estimator::TYPES[$estimator->type()] . 's.');
        }
    }
}
