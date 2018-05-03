<?php

namespace Rubix\Engine\Metrics;

use InvalidArgumentException;

class RMSError implements Regression
{
    /**
     * Calculate the root mean square error from the predictions.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $outcomes) : float
    {
        if (count($predictions) !== count($outcomes)) {
            throw new InvalidArgumentException('The number of outcomes must match the number of predictions.');
        }

        $error = 0;

        foreach ($predictions as $i => $prediction) {
            $error += ($outcomes[$i] - $prediction) ** 2;
        }

        return sqrt($error / count($outcomes));
    }

    /**
     * Should this metric be minimized?
     *
     * @return bool
     */
    public function minimize() : bool
    {
        return true;
    }
}
