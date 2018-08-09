<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\Clusterer;
use InvalidArgumentException;

class VMeasure implements Validation
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return array
     */
    public function range() : array
    {
        return [0, 1];
    }

    /**
     * Calculate the V score of a clustering. V Score is the harmonic mean of
     * homogeneity and completness.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(Estimator $estimator, Dataset $testing) : float
    {
        if (!$estimator instanceof Clusterer) {
            throw new InvalidArgumentException('This metric only works on'
                . ' clusterers.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This metric requires a labeled'
                . ' testing set.');
        }

        if ($testing->numRows() === 0) {
            return 0.0;
        }

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

        $homogeneity = $completeness = 0.0;

        foreach ($table[0] as $distribution) {
            $homogeneity += (max($distribution) + self::EPSILON)
                / (array_sum($distribution) + self::EPSILON);
        }

        foreach ($table[1] as $distribution) {
            $completeness += (max($distribution) + self::EPSILON)
                / (array_sum($distribution) + self::EPSILON);
        }

        $homogeneity /= count($table[0]);
        $completeness /= count($table[1]);

        return 2 * ($homogeneity * $completeness)
            / ($homogeneity + $completeness);
    }
}
