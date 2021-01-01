<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;

/**
 * Median Absolute Error
 *
 * Median Absolute Error (MAD) is a robust measure of error, similar to MAE, that ignores
 * highly erroneous predictions. Since MAD is a robust statistic, it works well even when
 * used to measure non-normal distributions.
 *
 * > **Note:** In order to maintain the convention of *maximizing* validation scores,
 * this metric outputs the negative of the original score.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MedianAbsoluteError implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return array{float,float}
     */
    public function range() : array
    {
        return [-INF, 0.0];
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
            EstimatorType::regressor(),
        ];
    }

    /**
     * Score a set of predictions.
     *
     * @param list<int|float> $predictions
     * @param list<int|float> $labels
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        PredictionAndLabelCountsAreEqual::with($predictions, $labels)->check();

        if (empty($predictions)) {
            return 0.0;
        }

        $errors = [];

        foreach ($predictions as $i => $prediction) {
            $errors[] = abs($labels[$i] - $prediction);
        }

        return -Stats::median($errors);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Median Absolute Error';
    }
}
