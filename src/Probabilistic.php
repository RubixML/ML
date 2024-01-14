<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

/**
 * Probabilistic
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Probabilistic extends Estimator
{
    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param Dataset $dataset
     * @return list<float[]>
     */
    public function proba(Dataset $dataset) : array;
}
