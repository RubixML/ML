<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;

use const Rubix\ML\EPSILON;

/**
 * R Squared
 *
 * The *coefficient of determination* or R Squared (R²) is the proportion of the variance in
 * the target labels that is explainable from the predictions. It gives an indication as to
 * how well the predictions approximate the labels.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RSquared implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return \Rubix\ML\Tuple{float,float}
     */
    public function range() : Tuple
    {
        return new Tuple(-INF, 1.0);
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
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        PredictionAndLabelCountsAreEqual::with($predictions, $labels)->check();

        if (empty($predictions)) {
            return 0.0;
        }

        $mean = Stats::mean($labels);

        $ssr = $sst = 0.0;

        foreach ($predictions as $i => $prediction) {
            $label = $labels[$i];

            $ssr += ($label - $prediction) ** 2;
            $sst += ($label - $mean) ** 2;
        }

        return 1.0 - ($ssr / ($sst ?: EPSILON));
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
        return 'R Squared';
    }
}
