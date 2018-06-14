<?php

namespace Rubix\ML\Metrics\Validation;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\Clusterer;

class VMeasure implements Clustering
{
    /**
     * Calculate the V score of a clustering. V Score is the harmonic mean of
     * homogeneity and completness.
     *
     * @param  \Rubix\ML\Clusterers\Clusterer  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return float
     */
    public function score(Clusterer $estimator, Labeled $testing) : float
    {
        $predictions = $estimator->predict($testing);

        $labels = $testing->labels();

        $clusters = array_unique($predictions);
        $classes = array_unique($labels);

        $table = [[], []];

        foreach ($clusters as $outcome) {
            $table[0][$outcome] = array_fill_keys($classes, 0);
        }

        foreach ($classes as $label) {
            $table[1][$label] = array_fill_keys($clusters, 0);
        }

        foreach ($predictions as $i => $outcome) {
            $table[0][$outcome][$labels[$i]] += 1;
        }

        foreach ($labels as $i => $class) {
            $table[1][$class][$predictions[$i]] += 1;
        }

        $homogeneity = 0.0;

        foreach ($table[0] as $distribution) {
            $homogeneity += max($distribution) / (array_sum($distribution)
                + self::EPSILON);
        }

        $homogeneity /= (count($table[0]) + self::EPSILON);

        $completeness = 0.0;

        foreach ($table[1] as $distribution) {
            $completeness += max($distribution) / (array_sum($distribution)
                + self::EPSILON);
        }

        $completeness /= (count($table[0]) + self::EPSILON);

        return 2 * ($homogeneity * $completeness)
            / ($homogeneity + $completeness);
    }
}
