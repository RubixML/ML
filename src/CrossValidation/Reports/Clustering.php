<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\Clusterer;

interface Clustering extends Report
{
    /**
     * Generate the clustering report.
     *
     * @param  \Rubix\ML\Clusterers\Clusterer  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return array
     */
    public function generate(Clusterer $estimator, Labeled $testing) : array;
}
