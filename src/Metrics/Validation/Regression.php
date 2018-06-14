<?php

namespace Rubix\Engine\Metrics\Validation;

use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Regressors\Regressor;

interface Regression extends Validation
{
    /**
     * Score a regressor using a labeled testing set.
     *
     * @param  \Rubix\Engine\Regressors\Regressor  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return float
     */
    public function score(Regressor $estimator, Labeled $testing) : float;
}
