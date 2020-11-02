<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

/**
 * Learner
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Learner extends Estimator
{
    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void;

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool;

    /**
     * Predict a single sample and return the result.
     *
     * @param (string|int|float)[] $sample
     * @return mixed
     */
    public function predictSample(array $sample);
}
