<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

/**
 * Completeness
 *
 * A ground-truth (requiring labels) clustering metric that measures the
 * ratio of samples in a class that are also members of the same cluster.
 * A cluster is said to be *complete* when all the samples ina class are
 * contained in a cluster.
 * 
 * References:
 * [1] A. Rosenberg et al. (2007). V-Measure: A conditional entropy-based
 * external cluster evaluation measure.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Completeness implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [0., 1.];
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
        if ($estimator->type() !== Estimator::CLUSTERER) {
            throw new InvalidArgumentException('This metric only works with'
                . ' clusterers.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This metric requires a labeled'
                . ' testing set.');
        }

        if ($testing->numRows() === 0) {
            return 0.;
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

        $score = 0.;

        foreach ($table as $distribution) {
            $score += max($distribution)
                / (array_sum($distribution) ?: self::EPSILON);
        }

        return $score / count($table);
    }
}
