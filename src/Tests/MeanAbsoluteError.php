<?php

namespace Rubix\Engine\Tests;

use InvalidArgumentException;

class MeanAbsoluteError extends Test
{
    /**
     * Calculate the mean absolute error of the predictions.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @return float
     */
    public function score(array $predictions, array $outcomes) : float
    {
        $error = 0.0;

        foreach ($predictions as $i => $prediction) {
            $error += abs($outcomes[$i] - $prediction);
        }

        $error = (1 / count($predictions)) * $error;

        $this->logger->log('Mean absolute error: ' . number_format($error, 5));

        return $error;
    }
}
