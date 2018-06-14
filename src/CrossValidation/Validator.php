<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;

interface Validator
{
    /**
    * Validate the estimator with the suppplied dataset and return a score.
    *
    * @param  \Rubix\ML\Estimator\Estimator  $estimator
    * @param  \Rubix\ML\Datasets\Labeled  $dataset
    * @return float
    */
   public function score(Estimator $estimator, Labeled $dataset) : float;
}
