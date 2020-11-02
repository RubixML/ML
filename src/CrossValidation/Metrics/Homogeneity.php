<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

use function count;

use const Rubix\ML\EPSILON;

/**
 * Homogeneity
 *
 * A ground-truth clustering metric that measures the ratio of samples in a cluster that
 * are also members of the same class. A cluster is said to be *homogeneous* when the
 * entire cluster is comprised of a single class of samples.
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
     * @return array{float,float}
     */
    public function range() : array
    {
        return [0.0, 1.0];
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\EstimatorType>
     */
    public function compatibility() : array
    {
        return [
            EstimatorType::clusterer(),
        ];
    }

    /**
     * Score a set of predictions.
     *
     * @param list<string|int> $predictions
     * @param list<string|int> $labels
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        $table = (new ContingencyTable())->generate($predictions, $labels);

        if (empty($table)) {
            return 0.0;
        }

        $score = 0.0;

        foreach ($table as $dist) {
            $score += max($dist) / (array_sum($dist) ?: EPSILON);
        }

        return $score / count($table);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Homogeneity';
    }
}
