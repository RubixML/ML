<?php

namespace Rubix\Engine\Metrics\Validation;

class Completeness implements Clustering
{
    /**
     * Calculate the completeness of a clustering.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        $clusters = array_unique($predictions);

        $table = [];

        foreach (array_unique($labels) as $class) {
            $table[$class] = array_fill_keys($clusters, 0);
        }

        foreach ($labels as $i => $class) {
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
