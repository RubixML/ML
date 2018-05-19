<?php

namespace Rubix\Engine\Metrics\Validation;

class MeanAbsoluteError implements Regression
{
    /**
     * Calculate the negative mean absolute error of the predictions.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        $error = 0.0;

        foreach ($predictions as $i => $prediction) {
            $error += abs($labels[$i] - $prediction->outcome());
        }

        return -($error / (count($predictions) + self::EPSILON));
    }
}
