<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Stringable;

interface Estimator extends Stringable
{
    /**
     * Return the estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType;

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array;

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array;

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return list<string|int|float>
     */
    public function predict(Dataset $dataset) : array;
}
