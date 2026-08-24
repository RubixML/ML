<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

use function count;

use const Rubix\ML\EPSILON;

/**
 * Class Purity
 *
 * A ground-truth clustering metric that measures the mean ratio of samples in a class
 * that are also members of the class' dominant cluster. A cluster is said to be
 * *complete* when all the samples in a class are contained in a single cluster.
 *
 * Note: This metric computes a purity score, not the conditional entropy-based
 * completeness measure defined by Rosenberg et al. (2007).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ClassPurity implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return \Rubix\ML\Tuple{float,float}
     */
    public function range() : Tuple
    {
        return new Tuple(0.0, 1.0);
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @internal
     *
     * @return list<EstimatorType>
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
        $table = (new ContingencyTable())->generate($labels, $predictions);

        if (count($table) === 0) {
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
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Class Purity';
    }
}
