<?php

namespace Rubix\Engine\CrossValidation;

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Estimator;

interface Validator
{
    /**
    * Validate the estimator with the suppplied dataset and return a score.
    *
    * @param  \Rubix\Engine\Estimator\Estimator  $estimator
    * @param  \Rubix\Engine\Datasets\Supervised  $dataset
    * @return float
    */
   public function score(Estimator $estimator, Supervised $dataset) : float;
}
