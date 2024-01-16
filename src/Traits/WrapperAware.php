<?php

namespace Rubix\ML\Traits;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Learner;

/**
 * Wrapper Aware
 *
 * This trait fulfills the requirements of the Wrapper interface and is suitable for most implementations.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 */
trait WrapperAware
{
    /**
     * The base estimator.
     *
     * @var Estimator
     */
    protected Estimator $base;

    /**
     * Return the base estimator instance.
     *
     * @return Estimator
     */
    public function base(): Estimator
    {
        return $this->base;
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return EstimatorType
     */
    public function type() : EstimatorType
    {
        return $this->base->type();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return $this->base->compatibility();
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        return $this->base->predict($dataset);
    }
}