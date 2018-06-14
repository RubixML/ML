<?php

namespace Rubix\Engine\Metrics\Validation;

use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Clusterers\Clusterer;

class Homogeneity implements Clustering
{
    /**
     * Calculate the homogeneity of a clustering.
     *
     * @param  \Rubix\Engine\Clusterers\Clusterer  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return float
     */
    public function score(Clusterer $estimator, Labeled $testing) : float
    {
        $predictions = $estimator->predict($testing);

        $labels = $testing->labels();

        $classes = array_unique($labels);

        $table = [];

        foreach (array_unique($predictions) as $outcome) {
            $table[$outcome] = array_fill_keys($classes, 0);
        }

        foreach ($predictions as $i => $outcome) {
            $table[$outcome][$labels[$i]] += 1;
        }

        $score = 0.0;

        foreach ($table as $distribution) {
            $score += max($distribution) / (array_sum($distribution)
                + self::EPSILON);
        }

        return $score / (count($table) + self::EPSILON);
    }
}
