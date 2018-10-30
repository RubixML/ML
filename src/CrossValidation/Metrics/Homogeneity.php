<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

/**
 * Homogeneity
 *
 * A ground-truth clustering metric that measures the ratio of samples in a
 * cluster that are also members of the same class. A cluster is said to be
 * *homogenous* when the entire cluster is comprised of a single class of
 * samples.
 * 
 * References:
 * [1] A. Rosenberg et al. (2007). V-Measure: A conditional entropy-based
 * external cluster evaluation measure.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Homogeneity implements Metric
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
     * Calculate the homogeneity of a clustering.
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

        if ($testing->numRows() < 1) {
            return 0.;
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

        $score = 0.;

        foreach ($table as $distribution) {
            $score += max($distribution)
                / (array_sum($distribution) ?: self::EPSILON);
        }

        return $score / count($table);
    }
}
