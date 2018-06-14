<?php

namespace Rubix\Engine\CrossValidation\Reports;

use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Clusterers\Clusterer;

interface Clustering extends Report
{
    /**
     * Generate the clustering report.
     *
     * @param  \Rubix\Engine\Clusterers\Clusterer  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return array
     */
    public function generate(Clusterer $estimator, Labeled $testing) : array;
}
