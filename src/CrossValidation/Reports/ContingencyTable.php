<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\Clusterer;

class ContingencyTable implements Clustering
{
    /**
     * Generate a contingency table for the clustering given a ground truth.
     *
     * @param  \Rubix\ML\Clusterers\Clusterer  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return array
     */
    public function generate(Clusterer $estimator, Labeled $testing) : array
    {
        $predictions = $estimator->predict($testing);

        $labels = $testing->labels();
        
        $classes = array_unique($labels);

        $table = [];

        foreach (array_unique($predictions) as $outcome) {
            $table[$outcome] = array_fill_keys($classes, 0);
        }

        foreach ($predictions as $i => $outcome) {
            $table[$outcome][$labels[$i]] += 1;
        }

        return $table;
    }
}
