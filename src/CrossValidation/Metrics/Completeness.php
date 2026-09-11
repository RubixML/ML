<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

use function array_sum;

/**
 * Completeness
 *
 * A ground-truth clustering metric that measures how well all the samples of a class
 * are grouped into a single cluster. A clustering is said to be *complete* when every
 * sample of a class is contained in one cluster. Formally, it is defined as one minus
 * the conditional entropy of the cluster assignments given the classes normalized by
 * the marginal entropy of the cluster assignments.
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
        $table = (new ContingencyTable())->generate($labels, $predictions);

        $conditional = $marginal = 0.0;
        $n = 0;

        $clusterCounts = [];

        foreach ($table as $dist) {
            $classSize = array_sum($dist);

            $n += $classSize;

            foreach ($dist as $cluster => $count) {
                if ($count === 0) {
                    continue;
                }

                $clusterCounts[$cluster] = ($clusterCounts[$cluster] ?? 0) + $count;

                $conditional += $count * log($classSize / $count);
            }
        }

        if ($n === 0) {
            return 0.0;
        }

        foreach ($clusterCounts as $clusterCount) {
            $marginal += $clusterCount * log($n / $clusterCount);
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
        return 'Completeness';
    }
}
