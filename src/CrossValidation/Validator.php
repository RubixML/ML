<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\Metrics\Validation;

interface Validator
{
    /**
    * Test the estimator with the suppplied dataset and return a score.
    *
    * @param  \Rubix\ML\Estimator\Estimator  $estimator
    * @param  \Rubix\ML\Datasets\Labeled  $dataset
    * @param  \Rubix\ML\CrossValidation\Metrics\Validation  $metric
    * @return float
    */
   public function test(Estimator $estimator, Labeled $dataset, Validation $metric) : float;
}
