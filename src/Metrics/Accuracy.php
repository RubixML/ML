<?php

namespace Rubix\Engine\Metrics;

use InvalidArgumentException;

class Accuracy implements Classification
{
    /**
     * Test the accuracy of the predictions.
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

        $n = count($predictions);
        $score = 0;

        foreach ($predictions as $i => $prediction) {
            if ($prediction === $outcomes[$i]) {
                $score++;
            }
        }

        return $score / ($n + self::EPSILON);
    }

    /**
     * Should this metric be minimized?
     *
     * @return bool
     */
    public function minimize() : bool
    {
        return false;
    }
}
