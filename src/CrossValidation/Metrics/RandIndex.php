<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Tensor\Matrix;
use Rubix\ML\Estimator;
use Rubix\ML\Other\Functions\Comb;
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
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        if (empty($predictions)) {
            return 0.0;
        }

        $n = count($predictions);

        if ($n !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        $report = new ContingencyTable();

        $table = Matrix::build($report->generate($predictions, $labels));

        $sigma = $table->map('\Rubix\ML\comb')->sum()->sum();

        $alpha = $table->sum()->map('\Rubix\ML\comb')->sum();
        $beta = $table->transpose()->sum()->map('\Rubix\ML\comb')->sum();

        $pHat = ($alpha * $beta) / comb($n);
        $mean = ($alpha + $beta) / 2.0;

        return ($sigma - $pHat) / ($mean - $pHat);
    }
}
