<?php

namespace Rubix\Engine\Metrics\Validation;

use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Classifiers\Classifier;

interface Classification extends Validation
{
    /**
     * Score a classifier using a labeled testing set.
     *
     * @param  \Rubix\Engine\Classifiers\Classifier  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return float
     */
    public function score(Classifier $estimator, Labeled $testing) : float;
}
