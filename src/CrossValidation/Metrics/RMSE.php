<?php

namespace Rubix\ML\CrossValidation\Metrics;

/**
 * RMSE
 *
 * The Root Mean Squared Error is equivalent to the average L2 loss.
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
     * @param array $predictions
     * @param array $labels
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        $score = parent::score($predictions, $labels);

        return -sqrt(-$score);
    }
}
