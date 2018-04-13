<?php

namespace Rubix\Engine\Tests;

use InvalidArgumentException;

class Accuracy extends Test
{
    /**
     * Test the accuracy of the predictions.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @return float
     */
    public function score(array $predictions, array $outcomes) : float
    {
        if (count($predictions) !== count($outcomes)) {
            throw new InvalidArgumentException('The number of outcomes must match the number of predictions.');
        }

        $accuracy = 0;

        foreach ($predictions as $i => $prediction) {
            if ($prediction === $outcomes[$i]) {
                $accuracy++;
            }
        }

        $accuracy /= count($outcomes);

        $this->logger->log('Model is ' . number_format($accuracy * 100, 2) . '% accurate');

        return $accuracy;
    }
}
