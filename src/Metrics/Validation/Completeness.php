<?php

namespace Rubix\Engine\Metrics\Validation;

use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Clusterers\Clusterer;

class Completeness implements Clustering
{
    /**
     * Calculate the completeness of a clustering.
     *
     * @param  \Rubix\Engine\Clusterers\Clusterer  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return float
     */
    public function score(Clusterer $estimator, Labeled $testing) : float
    {
        $predictions = $estimator->predict($testing);

        $clusters = array_unique($predictions);

        $table = [];

        foreach ($testing->possibleOutcomes() as $class) {
            $table[$class] = array_fill_keys($clusters, 0);
        }

        foreach ($testing->labels() as $i => $class) {
            $table[$class][$predictions[$i]] += 1;
        }

        $score = 0.0;

        foreach ($table as $distribution) {
            $score += max($distribution) / (array_sum($distribution)
                + self::EPSILON);
        }

        return $score / (count($table) + self::EPSILON);
    }
}
