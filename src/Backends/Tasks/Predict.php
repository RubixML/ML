<?php

namespace Rubix\ML\Backends\Tasks;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;

/**
 * Predict
 *
 * Routine to return the predictions made by an estimator on a dataset.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Predict extends Task
{
    /**
     * Return the predictions outputted by an estimator.
     *
     * @param Estimator $estimator
     * @param Dataset $dataset
     * @return list<string|int|float>
     */
    public static function predict(Estimator $estimator, Dataset $dataset) : array
    {
        return $estimator->predict($dataset);
    }

    /**
     * @param Estimator $estimator
     * @param Dataset $dataset
     */
    public function __construct(Estimator $estimator, Dataset $dataset)
    {
        parent::__construct([self::class, 'predict'], [$estimator, $dataset]);
    }
}
