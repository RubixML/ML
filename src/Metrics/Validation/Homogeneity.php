<?php

namespace Rubix\Engine\Metrics\Validation;

class Homogeneity implements Clustering
{
    /**
     * Calculate the homogeneity of a clustering.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
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
