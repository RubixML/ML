<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\Tensor\Matrix;
use Rubix\ML\Other\Functions\Comb;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;
use InvalidArgumentException;

/**
 * Rand Index
 *
 * The Adjusted Rand Index is a measure of similarity between the clustering
 * and some ground truth that is adjusted for chance. It considers all pairs
 * of samples that are assigned in the same or different clusters in the
 * predicted and empirical clusterings.
 *
 * References:
 * [1] W. M. Rand. (1971). Objective Criteria for the Evaluation of
 * Clustering Methods.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RandIndex implements Metric
{
    /**
     * Compute n choose k.
     *
     * @param int $n
     * @param int $k
     * @return int
     */
    public static function comb(int $n, int $k = 2) : int
    {
        return $k === 0 ? 1 : (int) (($n * self::comb($n - 1, $k - 1)) / $k);
    }

    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-1., 1.];
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

        $n = count($predictions);

        if ($n !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        $table = Matrix::build((new ContingencyTable())->generate($predictions, $labels));

        $sigma = $table->map([self::class, 'comb'])->sum()->sum();

        $alpha = $table->sum()->map([self::class, 'comb'])->sum();
        $beta = $table->transpose()->sum()->map([self::class, 'comb'])->sum();

        $pHat = ($alpha * $beta) / self::comb($n);
        $mean = ($alpha + $beta) / 2.;

        return ($sigma - $pHat) / ($mean - $pHat);
    }
}
