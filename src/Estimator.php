<?php

namespace Rubix\Engine;

interface Estimator
{
    /**
     * Train the model with a supervised dataset.
     *
     * @param  \Rubix\Engine\SupervisedDataset  $data
     * @return void
     */
    public function train(SupervisedDataset $data) : void;

    /**
     * Make a prediction of a given sample.
     *
     * @param  array  $sample
     * @return array
     */
    public function predict(array $sample) : array;
}
