<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\Metrics\Metric;

interface Validator
{
    /**
    * Test the estimator with the suppplied dataset and return a score.
    *
    * @param  \Rubix\ML\Estimator  $estimator
    * @param  \Rubix\ML\Datasets\Labeled  $dataset
    * @param  \Rubix\ML\CrossValidation\Metrics\Metric  $metric
    * @return float
    */
   public function test(Estimator $estimator, Labeled $dataset, Metric $metric) : float;
}
