<?php

namespace Rubix\Engine\Tests;

use InvalidArgumentException;

class StandardError extends Test
{
    /**
     * Calculate the standard error of the predictions.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @return float
     */
    public function score(array $predictions, array $outcomes) : float
    {
        $errors = [];

        foreach ($predictions as $i => $prediction) {
            $errors[] = abs($outcomes[$i] - $prediction);
        }

        $mean = array_sum($errors) / count($outcomes);

        $stddev = sqrt(array_reduce($errors, function ($carry, $error) use ($mean) {
            return $carry += ($error - $mean) ** 2;
        }, 0.0) / count($outcomes));

        $stderror = $stddev / sqrt(count($outcomes));

        $this->logger->log('Standard error: ' . number_format($stderror, 5));

        return $stderror;
    }
}
