<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
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
     * The estimator types that this metric is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            Estimator::CLUSTERER,
        ];
    }

    /**
     * Score a set of predictions.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        if (empty($predictions)) {
            return 0.;
        }

        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        $clusters = array_unique($predictions);
        $classes = array_unique($labels);

        $table = [];

        foreach ($clusters as $cluster) {
            $table[$cluster] = array_fill_keys($classes, 0);
        }

        foreach ($predictions as $i => $prediction) {
            $table[$prediction][$labels[$i]] += 1;
        }

        $score = 0.;

        foreach ($table as $dist) {
            $score += max($dist) / (array_sum($dist) ?: self::EPSILON);
        }

        return $score / count($table);
    }
}
