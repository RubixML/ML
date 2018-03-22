<?php

namespace Rubix\Engine;

interface Estimator
{
    /**
     * Train the model with a labeled dataset.
     *
     * @param  array  $data
     * @return self
     */
    public function train(array $data) : void;

    /**
     * Calculate the accuracy of the estimator with provided testing set.
     *
     * @return float
     */
    public function test(array $data) : float;

    /**
     * Make a prediction of a given sample.
     *
     * @param  array  $sample
     * @return array
     */
    public function predict(array $sample) : array;
}
