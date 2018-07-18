<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\Clusterer;
use InvalidArgumentException;

class Completeness implements Validation
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
     * Calculate the completeness score of a clustering.
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
            $score += max($distribution)
                / (array_sum($distribution) + self::EPSILON);
        }

        return $score / (count($table) + self::EPSILON);
    }
}
