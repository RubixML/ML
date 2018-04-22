<?php

namespace Rubix\Engine\Tests;

use InvalidArgumentException;

class RMSError extends Test
{
    /**
     * Calculate the root mean square error from the predictions.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @return float
     */
    public function score(array $predictions, array $outcomes) : float
    {
        $error = 0.0;

        foreach ($predictions as $i => $prediction) {
            $error += ($outcomes[$i] - $prediction) ** 2;
        }

        $rmse = sqrt((1 / count($predictions)) * $error);

        $this->logger->log('RMS error: ' . number_format($rmse, 5));

        return $rmse;
    }
}
