<?php

namespace Rubix\Engine\Metrics;

use InvalidArgumentException;

class MeanAbsoluteError implements Error
{
    /**
     * Calculate the mean absolute error of the predictions.
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

        $error = 0.0;

        foreach ($predictions as $i => $prediction) {
            $error += abs($outcomes[$i] - $prediction);
        }

        return $error / count($predictions);
    }
}
