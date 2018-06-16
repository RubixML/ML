<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;

interface Validator
{
    /**
    * Test the estimator with the suppplied dataset and return a score.
    *
    * @param  \Rubix\ML\Estimator\Estimator  $estimator
    * @param  \Rubix\ML\Datasets\Labeled  $dataset
    * @return float
    */
   public function test(Estimator $estimator, Labeled $dataset) : float;
}
