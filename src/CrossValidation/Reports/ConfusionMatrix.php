<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\Classifier;
use InvalidArgumentException;

class ConfusionMatrix implements Report
{
    /**
     * The classes to compare in the matrix.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * @param  array|null  $labels
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
     * @param  \Runix\ML\Datasets\Labeled  $testing
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(Estimator $estimator, Labeled $testing) : array
    {
        if (!$estimator instanceof Classifier) {
            throw new InvalidArgumentException('This report only works on'
                . ' classifiers.');
        }

        $predictions = $estimator->predict($testing);

        $labels = $testing->labels();

        if (!isset($this->classes)) {
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
                $matrix[$outcome][$outcome] += 1;
            } else {
                $matrix[$outcome][$labels[$i]] += 1;
            }
        }

        return $matrix;
    }
}
