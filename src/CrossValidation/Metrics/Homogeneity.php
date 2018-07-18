<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\Clusterer;
use InvalidArgumentException;

class Homogeneity implements Validation
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
     * Calculate the homogeneity of a clustering.
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
