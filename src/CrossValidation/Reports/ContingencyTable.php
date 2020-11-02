<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Report;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;

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
class ContingencyTable implements ReportGenerator
{
    /**
     * The estimator types that this report is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\EstimatorType>
     */
    public function compatibility() : array
    {
        return [
            EstimatorType::clusterer(),
        ];
    }

    /**
     * Generate the report.
     *
     * @param list<string|int> $predictions
     * @param list<string|int> $labels
     * @return \Rubix\ML\Report
     */
    public function generate(array $predictions, array $labels) : Report
    {
        PredictionAndLabelCountsAreEqual::with($predictions, $labels)->check();

        $classes = array_unique($labels);
        $clusters = array_unique($predictions);

        $table = array_fill_keys($clusters, array_fill_keys($classes, 0));

        foreach ($labels as $i => $class) {
            ++$table[$predictions[$i]][$class];
        }

        return new Report($table);
    }
}
