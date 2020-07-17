<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use InvalidArgumentException;

use function count;
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
class ConfusionMatrix implements Report
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
     * @return \Rubix\ML\EstimatorType[]
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
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @throws \InvalidArgumentException
     * @return array[]
     */
    public function generate(array $predictions, array $labels) : array
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('Number of predictions'
                . ' and labels must be equal.');
        }

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

        return $matrix;
    }
}
