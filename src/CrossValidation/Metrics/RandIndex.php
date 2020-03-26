<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Tensor\Matrix;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;
use InvalidArgumentException;

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
     * @return float[]
     */
    public function range() : array
    {
        return [-1.0, 1.0];
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @return \Rubix\ML\EstimatorType[]
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
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('Number of predictions'
                . ' and labels must be equal.');
        }

        if (empty($predictions)) {
            return 0.0;
        }

        $table = Matrix::build((new ContingencyTable())->generate($labels, $predictions));

        $sigma = $table->map([self::class, 'comb2'])->sum()->sum();

        $alpha = $table->sum()->map([self::class, 'comb2'])->sum();
        $beta = $table->transpose()->sum()->map([self::class, 'comb2'])->sum();

        $pHat = ($alpha * $beta) / self::comb2(count($predictions));
        $mean = ($alpha + $beta) / 2.0;

        return ($sigma - $pHat) / ($mean - $pHat);
    }
}
