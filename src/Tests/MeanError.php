<?php

namespace Rubix\Engine\Tests;

use InvalidArgumentException;

class MeanError extends Test
{
    /**
     * Calculate the mean error of the predictions.
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

        $error /= count($outcomes);

        $this->logger->log('Mean error: ' . number_format($error, 5));

        return $error;
    }
}
