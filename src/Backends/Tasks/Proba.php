<?php

namespace Rubix\ML\Backends\Tasks;

use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;

/**
 * Proba
 *
 * Routine to return the joint probabilities made by an estimator on a dataset.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Proba extends Task
{
    /**
     * Return the probabilities outputted by the estimator.
     *
     * @param Probabilistic $estimator
     * @param Dataset $dataset
     * @return list<float[]>
     */
    public static function proba(Probabilistic $estimator, Dataset $dataset) : array
    {
        return $estimator->proba($dataset);
    }

    /**
     * @param Probabilistic $estimator
     * @param Dataset $dataset
     */
    public function __construct(Probabilistic $estimator, Dataset $dataset)
    {
        parent::__construct([self::class, 'proba'], [$estimator, $dataset]);
    }
}
