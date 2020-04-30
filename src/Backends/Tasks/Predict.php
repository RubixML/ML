<?php

namespace Rubix\ML\Backends\Tasks;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;

class Predict extends Task
{
    /**
     * Return the predictions outputted by an estimator.
     *
     * @param \Rubix\ML\Estimator $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return (int|float|string)[]
     */
    public static function predict(Estimator $estimator, Dataset $dataset) : array
    {
        return $estimator->predict($dataset);
    }

    /**
     * @param \Rubix\ML\Estimator $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function __construct(Estimator $estimator, Dataset $dataset)
    {
        parent::__construct([self::class, 'predict'], [$estimator, $dataset]);
    }
}
