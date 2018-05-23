<?php

namespace Rubix\Engine\CrossValidation\Reports;

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
     * @param  array  $predictions
     * @param  array  $labels
     * @return array
     */
    public function generate(array $predictions, array $labels) : array
    {
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
