<?php

namespace Rubix\ML\CrossValidation\Metrics;

interface RegressionMetric extends Metric
{
    /**
     * Score a set of continuous predictions and their ground-truth labels.
     *
     * @param list<int|float> $predictions
     * @param list<int|float> $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float;
}
