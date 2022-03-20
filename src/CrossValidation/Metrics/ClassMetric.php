<?php

namespace Rubix\ML\CrossValidation\Metrics;

interface ClassMetric extends DiscreteMetric
{
    /**
     * Score a set of class predictions and their ground-truth labels.
     *
     * @param list<string|int> $predictions
     * @param list<string|int> $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float;
}
