<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\Metrics\Metric;

interface Validator
{
    /**
     * Return the validation scores computed at last test time.
     * 
     * @return array|null
     */
    public function scores() : ?array;

    /**
    * Test the estimator with the supplied dataset and return a score.
    *
    * @param  \Rubix\ML\Estimator  $estimator
    * @param  \Rubix\ML\Datasets\Labeled  $dataset
    * @param  \Rubix\ML\CrossValidation\Metrics\Metric  $metric
    * @return float
    */
   public function test(Estimator $estimator, Labeled $dataset, Metric $metric) : float;
}
