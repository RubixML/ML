<?php

namespace Rubix\ML;

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
