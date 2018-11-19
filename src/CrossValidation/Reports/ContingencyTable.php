<?php

namespace Rubix\ML\CrossValidation\Reports;

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
     * Generate the report.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(array $predictions, array $labels) : array
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        $clusters = array_unique($predictions);
        $classes = array_unique($labels);

        $table = [];

        foreach ($clusters as $cluster) {
            $table[$cluster] = array_fill_keys($classes, 0);
        }

        foreach ($predictions as $i => $prediction) {
            $table[$prediction][$labels[$i]]++;
        }

        return $table;
    }
}
