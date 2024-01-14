<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Learner;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Stringable;

interface Validator extends Stringable
{
    /**
     * Test the estimator with the supplied dataset and return a validation score.
     *
     * @param Learner $estimator
     * @param Labeled $dataset
     * @param Metric $metric
     * @return float
     */
    public function test(Learner $estimator, Labeled $dataset, Metric $metric) : float;
}
