<?php

namespace Rubix\Engine;

interface Estimator
{
    /**
     * Train the model with a labeled dataset.
     *
     * @param  array  $samples
     * @param  array  $outcomes
     * @return void
     */
    public function train(array $samples, array $outcomes) : void;

    /**
     * Make a prediction of a given sample.
     *
     * @param  array  $sample
     * @return array
     */
    public function predict(array $sample) : array;
}
