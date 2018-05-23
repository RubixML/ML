<?php

namespace Rubix\Engine\Metrics\Validation;

class MeanSquaredError implements Regression
{
    /**
     * Calculate the negative mean squared error of the predictions.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        $error = 0.0;

        foreach ($predictions as $i => $outcome) {
            $error += ($labels[$i] - $outcome) ** 2;
        }

        return -($error / (count($predictions) + self::EPSILON));
    }
}
