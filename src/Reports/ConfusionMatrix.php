<?php

namespace Rubix\ML\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

/**
 * Confusion Matrix
 *
 * A Confusion Matrix is a table that visualizes the true positives, false,
 * positives, true negatives, and false negatives of a Classifier. The name
 * stems from the fact that the matrix makes it easy to see the classes that the
 * Classifier might be confusing.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ConfusionMatrix implements Report
{
    /**
     * The classes to compare in the matrix.
     *
     * @var array|null
     */
    protected $classes;

    /**
     * @param  array|null  $classes
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(?array $classes = null)
    {
        if (is_array($classes)) {
            $classes = array_unique($classes);

            foreach ($classes as $class) {
                if (!is_string($class) and !is_int($class)) {
                    throw new InvalidArgumentException('Class type must be a'
                        . ' string or integer, ' . gettype($class) . ' found.');
                }
            }
        }

        $this->classes = $classes;
    }

    /**
     * Generate the report.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(Estimator $estimator, Dataset $testing) : array
    {
        if ($estimator->type() !== Estimator::CLASSIFIER and $estimator->type() !== Estimator::DETECTOR) {
            throw new InvalidArgumentException('This report only works with'
                . ' classifiers and anomaly detectors.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This report requires a'
                . ' Labeled testing set.');
        }

        if ($testing->numRows() === 0) {
            throw new InvalidArgumentException('Testing set must contain at'
                . ' least one sample.');
        }

        $predictions = $estimator->predict($testing);

        $labels = $testing->labels();

        if (is_null($this->classes)) {
            $classes = array_unique(array_merge($predictions, $labels));
        } else {
            $classes = $this->classes;
        }

        $matrix = array_fill_keys($classes, array_fill_keys($classes, 0));

        foreach ($predictions as $i => $outcome) {
            if (!isset($matrix[$outcome])) {
                continue 1;
            }

            $matrix[$outcome][$labels[$i]]++;
        }

        return $matrix;
    }
}
