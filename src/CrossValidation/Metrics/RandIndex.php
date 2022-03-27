<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Tensor\Matrix;
use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

use function count;
use function Rubix\ML\comb;

/**
 * Rand Index
 *
 * The Adjusted Rand Index is a measure of similarity between a clustering and some
 * ground-truth that is adjusted for chance. It considers all pairs of samples that are
 * assigned in the same or different clusters in the predicted and empirical clusterings.
 *
 * References:
 * [1] W. M. Rand. (1971). Objective Criteria for the Evaluation of Clustering Methods.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RandIndex implements Metric
{
    /**
     * Compute n choose 2.
     *
     * @internal
     *
     * @param int $n
     * @return int
     */
    public static function comb2(int $n) : int
    {
        return comb($n, 2);
    }

    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return \Rubix\ML\Tuple{float,float}
     */
    public function range() : Tuple
    {
        return new Tuple(-1.0, 1.0);
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

        $table = Matrix::build($table->toArray());

        $sigma = $table->map([self::class, 'comb2'])->sum()->sum();

        $alpha = $table->sum()->map([self::class, 'comb2'])->sum();
        $beta = $table->transpose()->sum()->map([self::class, 'comb2'])->sum();

        $pHat = ($alpha * $beta) / self::comb2(count($predictions));
        $mean = ($alpha + $beta) / 2.0;

        return ($sigma - $pHat) / ($mean - $pHat);
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
        return 'Rand Index';
    }
}
