<?php

namespace Rubix\ML\CrossValidation\Metrics;

interface ProbabilisticMetric extends DiscreteMetric
{
    /**
     * Return the validation score of a set of class probabilities with their ground-truth labels.
     *
     * @param list<array<float>> $probabilities
     * @param list<string|int> $labels
     * @return float
     */
    public function score(array $probabilities, array $labels) : float;
}
