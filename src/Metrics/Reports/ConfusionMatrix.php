<?php

namespace Rubix\Engine\Metrics\Reports;

use InvalidArgumentException;

class ConfusionMatrix implements Report
{
    /**
     * The labels to compare in the matrix.
     *
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * @param  array  $labels
     * @return void
     */
    public function __construct(array $labels)
    {
        $this->labels = $labels;
    }

    /**
     * @param  array  $predictions
     * @param  array  $outcomes
     * @throws \InvalidArgumentException
     * @return void
     */
    public function generate(array $predictions, array $outcomes) : array
    {
        if (count($predictions) !== count($outcomes)) {
            throw new InvalidArgumentException('The number of outcomes must match the number of predictions.');
        }

        $matrix = [];

        foreach ($this->labels as $label) {
            $matrix[$label] = array_fill_keys($this->labels, 0);
        }

        foreach ($outcomes as $i => $outcome) {
            if (!isset($matrix[$outcome]) || !isset($matrix[$predictions[$i]])) {
                continue 1;
            }

            if ($predictions[$i] === $outcome) {
                $matrix[$outcome][$outcome] += 1;
            } else {
                $matrix[$outcome][$predictions[$i]] += 1;
            }
        }

        return $matrix;
    }
}
