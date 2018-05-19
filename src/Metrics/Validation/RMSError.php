<?php

namespace Rubix\Engine\Metrics\Validation;

class RMSError implements Regression
{
    /**
     * Calculate the negative root mean square error from the predictions.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        $error = 0.0;

        foreach ($predictions as $i => $prediction) {
            $error += ($labels[$i] - $prediction->outcome()) ** 2;
        }

        return -sqrt($error / (count($predictions) + self::EPSILON));
    }
}
