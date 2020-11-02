<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Report;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;

use function is_null;

/**
 * Confusion Matrix
 *
 * A Confusion Matrix is a square matrix (table) that visualizes the true positives, false positives,
 * true negatives, and false negatives of a set of class predictions and their corresponding labels.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ConfusionMatrix implements ReportGenerator
{
    /**
     * The classes to include in the report.
     *
     * @var (string|int)[]|null
     */
    protected $classes;

    /**
     * @param (string|int)[]|null $classes
     */
    public function __construct(?array $classes = null)
    {
        $this->classes = $classes;
    }

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
            EstimatorType::classifier(),
            EstimatorType::anomalyDetector(),
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

        if (is_null($this->classes)) {
            $classes = array_unique(array_merge($predictions, $labels));
        } else {
            $classes = $this->classes;
        }

        $matrix = array_fill_keys($classes, array_fill_keys($classes, 0));

        $classes = array_flip($classes);

        foreach ($predictions as $i => $prediction) {
            if (isset($classes[$prediction])) {
                $label = $labels[$i];

                if (isset($classes[$label])) {
                    ++$matrix[$prediction][$label];
                }
            }
        }

        return new Report($matrix);
    }
}
