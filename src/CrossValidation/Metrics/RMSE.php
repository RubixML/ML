<?php

namespace Rubix\ML\CrossValidation\Metrics;

/**
 * RMSE
 *
 * The Root Mean Squared Error (RMSE) is equivalent to the standard deviation of the
 * error residuals in a regression problem. Since RMSE is just the square root of the
 * MSE, RMSE is also sensitive to outliers because larger errors have a
 * disproportionately large effect on the score.
 *
 * > **Note:** In order to maintain the convention of *maximizing* validation scores,
 * this metric outputs the negative of the original score.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RMSE extends MeanSquaredError
{
    /**
     * Score a set of predictions.
     *
     * @param (int|float)[] $predictions
     * @param (int|float)[] $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        return -sqrt(-parent::score($predictions, $labels));
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
        return 'RMSE';
    }
}
