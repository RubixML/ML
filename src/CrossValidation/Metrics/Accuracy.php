<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;

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
     * @return list<\Rubix\ML\EstimatorType>
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
     * @param list<string|int> $predictions
     * @param list<string|int> $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        PredictionAndLabelCountsAreEqual::with($predictions, $labels)->check();

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

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Accuracy';
    }
}
