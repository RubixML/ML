<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;

interface Report
{
    const EPSILON = 1e-8;

    /**
     * Generate the report.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset $testing
     * @return array
     */
    public function generate(Estimator $estimator, Dataset $testing) : array;
}
