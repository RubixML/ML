<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Estimator;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\CrossValidation\Metrics\Metric;
use InvalidArgumentException;

use function get_class;
use function in_array;

class EstimatorIsCompatibleWithMetric
{
    /**
     * Perform a check of the specification.
     *
     * @param \Rubix\ML\Estimator $estimator
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @throws \InvalidArgumentException
     */
    public static function check(Estimator $estimator, Metric $metric) : void
    {
        if (!in_array($estimator->type(), $metric->compatibility())) {
            throw new InvalidArgumentException(
                Params::shortName(get_class($metric))
                    . ' is only compatible with '
                    . implode(', ', $metric->compatibility())
                    . " estimator types, {$estimator->type()} given."
            );
        }
    }
}
