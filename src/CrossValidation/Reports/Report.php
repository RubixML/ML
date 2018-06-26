<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;

interface Report
{
    const EPSILON = 1e-8;

    /**
     * Generate the report.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Runix\ML\Datasets\Labeled  $testing
     * @return array
     */
    public function generate(Estimator $estimator, Labeled $testing) : array;
}
