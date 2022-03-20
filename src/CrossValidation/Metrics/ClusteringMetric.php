<?php

namespace Rubix\ML\CrossValidation\Metrics;

interface ClusteringMetric extends Metric
{
    /**
     * Score a set of sample clusterings and their known class labels.
     *
     * @param list<int> $predictions
     * @param list<string|int> $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float;
}
