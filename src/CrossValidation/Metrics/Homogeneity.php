<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

use function array_sum;

/**
 * Homogeneity
 *
 * A ground-truth clustering metric that measures how well each cluster is comprised of
 * samples from a single class. A clustering is said to be *homogeneous* when all of its
 * clusters contain only samples of a single class. Formally, it is defined as one minus
 * the conditional entropy of the classes given the cluster assignments normalized by the
 * marginal entropy of the classes.
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
     * @return Tuple<float,float>
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
        $table = (new ContingencyTable())->generate($predictions, $labels);

        $conditional = $marginal = 0.0;
        $n = 0;

        $classCounts = [];

        foreach ($table as $dist) {
            $clusterSize = array_sum($dist);

            $n += $clusterSize;

            foreach ($dist as $class => $count) {
                if ($count === 0) {
                    continue;
                }

                $classCounts[$class] = ($classCounts[$class] ?? 0) + $count;

                $conditional += $count * log($clusterSize / $count);
            }
        }

        if ($n === 0) {
            return 0.0;
        }

        foreach ($classCounts as $classCount) {
            $marginal += $classCount * log($n / $classCount);
        }

        if ($marginal === 0.0) {
            return 1.0;
        }

        $score = max(0.0, min(1.0, 1.0 - $conditional / $marginal));

        return $score;
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
        return 'Homogeneity';
    }
}
