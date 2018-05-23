<?php

namespace Rubix\Engine\CrossValidation;

use Rubix\Engine\Estimator;
use Rubix\Engine\Datasets\Labeled;

interface Validator
{
    /**
    * Validate the estimator with the suppplied dataset and return a score.
    *
    * @param  \Rubix\Engine\Estimator\Estimator  $estimator
    * @param  \Rubix\Engine\Datasets\Labeled  $dataset
    * @return float
    */
   public function score(Estimator $estimator, Labeled $dataset) : float;
}
