<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Metrics\Validation\Validation;

interface Validator
{
    /**
    * Test the estimator with the suppplied dataset and return a score.
    *
    * @param  \Rubix\ML\Estimator\Estimator  $estimator
    * @param  \Rubix\ML\Datasets\Labeled  $dataset
    * @param  \Rubix\ML\Metrics\Validation\Validation  $metric
    * @return float
    */
   public function test(Estimator $estimator, Labeled $dataset, Validation $metric) : float;
}
