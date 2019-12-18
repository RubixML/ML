<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use InvalidArgumentException;

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
     * @return float[]
     */
    public function range() : array
    {
        return [-100., 0.];
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            Estimator::REGRESSOR,
        ];
    }

    /**
     * Score a set of predictions.
     *
     * @param (int|float)[] $predictions
     * @param (int|float)[] $labels
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        if (empty($predictions)) {
            return 0.;
        }

        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        $error = 0.;

        foreach ($predictions as $i => $prediction) {
            $label = $labels[$i];

            $error += 100. * abs(($prediction - $label)
                / ((abs($label) + abs($prediction)) ?: EPSILON));
        }

        return -($error / count($predictions));
    }
}
