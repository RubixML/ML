<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;

use function count;

use const Rubix\ML\EPSILON;

/**
 * SMAPE
 *
 * *Symmetric Mean Absolute Percentage Error* (SMAPE) is a scale-independent regression
 * metric that expresses the relative error of a set of predictions and their labels as a
 * percentage. It is an improvement over the non-symmetric MAPE in that it is both upper
 * and lower bounded.
 *
 * References:
 * [1] V. Kreinovich. et al. How to Estimate Forecasting Quality: A System Motivated
 * Derivation of Symmetric Mean Absolute Percentage Error (SMAPE) and Other Similar
 * Characteristics.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SMAPE implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return \Rubix\ML\Tuple{float,float}
     */
    public function range() : Tuple
    {
        return new Tuple(-100.0, 0.0);
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

        $error = 0.0;

        foreach ($predictions as $i => $prediction) {
            $label = $labels[$i];

            $error += 100.0 * abs(($prediction - $label)
                / ((abs($label) + abs($prediction)) ?: EPSILON));
        }

        return -($error / count($predictions));
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
        return 'SMAPE';
    }
}
