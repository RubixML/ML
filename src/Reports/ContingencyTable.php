<?php

namespace Rubix\ML\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\Clusterer;
use InvalidArgumentException;

/**
 * Contingency Table
 *
 * A Contingency Table is used to display the frequency distribution of class
 * labels among a clustering of samples.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ContingencyTable implements Report
{
    /**
     * Generate a contingency table for the clustering given a ground truth.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(Estimator $estimator, Dataset $testing) : array
    {
        if (!$estimator instanceof Clusterer) {
            throw new InvalidArgumentException('This report only works on'
                . ' clusterers.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This report requires a'
                . ' Labeled testing set.');
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
