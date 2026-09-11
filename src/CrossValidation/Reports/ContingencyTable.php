<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Report;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;
use Rubix\ML\Set;

use function array_fill_keys;

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
     * @return list<EstimatorType>
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
     * @return Report
     */
    public function generate(array $predictions, array $labels) : Report
    {
        PredictionAndLabelCountsAreEqual::with($predictions, $labels)->check();

        $clusters = new Set(...$predictions);
        $classes = new Set(...$labels);

        $table = array_fill_keys($clusters->toArray(), array_fill_keys($classes->toArray(), 0));

        foreach ($predictions as $i => $prediction) {
            ++$table[$prediction][$labels[$i]];
        }

        return new Report($table);
    }
}
