<?php

namespace Rubix\ML\Exceptions;

use Rubix\ML\Estimator;
use Rubix\ML\CrossValidation\Metrics\Metric;

class EstimatorIncompatibleWithMetric extends InvalidArgumentException
{
    /**
     * @param \Rubix\ML\Estimator $estimator
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     */
    public function __construct(Estimator $estimator, Metric $metric)
    {
        parent::__construct("{$metric} is not compatible with {$estimator->type()}s.");
    }
}
