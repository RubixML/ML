<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\Clusterer;
use InvalidArgumentException;

class ContingencyTable implements Report
{
    /**
     * Generate a contingency table for the clustering given a ground truth.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Runix\ML\Datasets\Labeled  $testing
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(Estimator $estimator, Labeled $testing) : array
    {
        if (!$estimator instanceof Clusterer) {
            throw new InvalidArgumentException('This report only works on'
                . ' clusterers.');
        }

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
