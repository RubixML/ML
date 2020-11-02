<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

use function count;

use const Rubix\ML\EPSILON;

/**
 * Completeness
 *
 * A ground-truth clustering metric that measures the ratio of samples in a class that
 * are also members of the same cluster. A cluster is said to be *complete* when all the
 * samples in a class are contained in a cluster.
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
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        $table = (new ContingencyTable())->generate($labels, $predictions);

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
        return 'Completeness';
    }
}
