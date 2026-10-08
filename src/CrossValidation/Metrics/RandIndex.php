<?php

namespace Rubix\ML\CrossValidation\Metrics;

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
     * @return Tuple<float,float>
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
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        $n = count($predictions);

        if ($n < 2) {
            return 1.0;
        }

        $table = (new ContingencyTable())->generate($labels, $predictions)->toArray();

        $sigma = $alpha = $beta = 0;

        foreach ($table as $row) {
            $rowSum = 0;

            foreach ($row as $count) {
                $sigma += self::comb2($count);
                $rowSum += $count;
            }

            $alpha += self::comb2($rowSum);
        }

        $columns = [];

        foreach ($table as $row) {
            foreach ($row as $label => $count) {
                $columns[$label] = ($columns[$label] ?? 0) + $count;
            }
        }

        foreach ($columns as $count) {
            $beta += self::comb2($count);
        }

        $pHat = ($alpha * $beta) / self::comb2($n);
        $mean = ($alpha + $beta) / 2.0;

        if ($mean == $pHat) {
            return 1.0;
        }

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
