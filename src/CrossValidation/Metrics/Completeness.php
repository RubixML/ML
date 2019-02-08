<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
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
     * @param array $predictions
     * @param array $labels
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

        foreach ($classes as $class) {
            $table[$class] = array_fill_keys($clusters, 0);
        }

        foreach ($labels as $i => $label) {
            $table[$label][$predictions[$i]] += 1;
        }

        $score = 0.;

        foreach ($table as $dist) {
            $score += max($dist) / (array_sum($dist) ?: self::EPSILON);
        }

        return $score / count($table);
    }
}
