<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use InvalidArgumentException;

use function count;

/**
 * Accuracy
 *
 * A quick and simple classification and anomaly detection metric defined as the
 * number of true positives over the number of samples in the testing set. Since
 * Accuracy gives equal weight to false positives and false negatives, it is *not* a
 * good metric for datasets with a highly imbalanced distribution of labels.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Accuracy implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [0.0, 1.0];
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @return \Rubix\ML\EstimatorType[]
     */
    public function compatibility() : array
    {
        return [
            EstimatorType::classifier(),
            EstimatorType::anomalyDetector(),
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

        $score = 0.0;

        foreach ($predictions as $i => $prediction) {
            if ($prediction == $labels[$i]) {
                ++$score;
            }
        }

        return $score / count($predictions);
    }
}
