<?php

namespace Rubix\ML\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\Classifier;
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
     * @return void
     */
    public function __construct(?array $classes = null)
    {
        $this->classes = $classes;
    }

    /**
     * Generate a confusion matrix.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(Estimator $estimator, Dataset $testing) : array
    {
        if (!$estimator instanceof Classifier) {
            throw new InvalidArgumentException('This report only works on'
                . ' classifiers.');
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

        $matrix = [];

        foreach ($classes as $class) {
            $matrix[$class] = array_fill_keys($classes, 0);
        }

        foreach ($predictions as $i => $outcome) {
            if (!isset($matrix[$outcome]) or !isset($matrix[$labels[$i]])) {
                continue 1;
            }

            if ($outcome === $labels[$i]) {
                $matrix[$outcome][$outcome]++;
            } else {
                $matrix[$outcome][$labels[$i]]++;
            }
        }

        return $matrix;
    }
}
