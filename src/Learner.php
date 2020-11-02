<?php

namespace Rubix\ML;

/**
 * Learner
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Learner extends Trainable, Estimator
{
    /**
     * Predict a single sample and return the result.
     *
     * @param (string|int|float)[] $sample
     * @return mixed
     */
    public function predictSample(array $sample);
}
