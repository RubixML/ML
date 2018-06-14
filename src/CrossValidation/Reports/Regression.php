<?php

namespace Rubix\Engine\CrossValidation\Reports;

use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Regressors\Regressor;

interface Regression extends Report
{
    /**
     * Generate the regression report.
     *
     * @param  \Rubix\Engine\Regressors\Regressor  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return array
     */
    public function generate(Regressor $estimator, Labeled $testing) : array;
}
