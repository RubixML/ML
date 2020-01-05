<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use InvalidArgumentException;

/**
 * Contingency Table
 *
 * A Contingency Table is used to display the frequency distribution of class labels among
 * a clustering. It is similar to a Confusion Matrix but uses the labels to establish
 * ground-truth for a clustering problem instead.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ContingencyTable implements Report
{
    /**
     * The estimator types that this report is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            Estimator::CLUSTERER,
        ];
    }

    /**
     * Generate the report.
     *
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @throws \InvalidArgumentException
     * @return array[]
     */
    public function generate(array $predictions, array $labels) : array
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        $classes = array_unique($labels);
        $clusters = array_unique($predictions);

        $table = array_fill_keys($clusters, array_fill_keys($classes, 0));

        foreach ($labels as $i => $class) {
            ++$table[$predictions[$i]][$class];
        }

        return $table;
    }
}
