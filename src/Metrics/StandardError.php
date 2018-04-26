<?php

namespace Rubix\Engine\Metrics;

use InvalidArgumentException;

class StandardError implements Error
{
    /**
     * Calculate the standard error of the mean from a set of predictions.
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

        $errors = [];

        foreach ($predictions as $i => $prediction) {
            $errors[] = abs($outcomes[$i] - $prediction);
        }

        $mean = array_sum($errors) / count($outcomes);

        $stddev = sqrt(array_reduce($errors, function ($carry, $error) use ($mean) {
            return $carry += ($error - $mean) ** 2;
        }, 0.0) / count($outcomes));

        return $stddev / sqrt(count($outcomes));
    }
}
