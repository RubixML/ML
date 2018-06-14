<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Regressor;

interface Regression extends Report
{
    /**
     * Generate the regression report.
     *
     * @param  \Rubix\ML\Regressors\Regressor  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return array
     */
    public function generate(Regressor $estimator, Labeled $testing) : array;
}
