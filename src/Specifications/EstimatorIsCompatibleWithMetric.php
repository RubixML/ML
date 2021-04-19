<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Estimator;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Exceptions\EstimatorIncompatibleWithMetric;

use function in_array;

/**
 * @internal
 */
class EstimatorIsCompatibleWithMetric extends Specification
{
    /**
     * The estimator.
     *
     * @var \Rubix\ML\Estimator
     */
    protected \Rubix\ML\Estimator $estimator;

    /**
     * The validation metric.
     *
     * @var \Rubix\ML\CrossValidation\Metrics\Metric
     */
    protected \Rubix\ML\CrossValidation\Metrics\Metric $metric;

    /**
     * Build a specification object with the given arguments.
     *
     * @param \Rubix\ML\Estimator $estimator
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @return self
     */
    public static function with(Estimator $estimator, Metric $metric) : self
    {
        return new self($estimator, $metric);
    }

    /**
     * @param \Rubix\ML\Estimator $estimator
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     */
    public function __construct(Estimator $estimator, Metric $metric)
    {
        $this->estimator = $estimator;
        $this->metric = $metric;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function check() : void
    {
        if (!in_array($this->estimator->type(), $this->metric->compatibility())) {
            throw new EstimatorIncompatibleWithMetric($this->estimator, $this->metric);
        }
    }
}
