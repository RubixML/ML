<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Stringable;

interface ProbabilisticMetric extends Stringable
{
    /**
     * Return a tuple of the min and max score for this metric.
     *
     * @return \Rubix\ML\Tuple{float,float}
     */
    public function range() : Tuple;

    /**
     * Return the validation score of a set of probabilities with their ground-truth labels.
     *
     * @param list<array<string|int,float>> $probabilities
     * @param list<string|int> $labels
     * @return float
     */
    public function score(array $probabilities, array $labels) : float;
}
