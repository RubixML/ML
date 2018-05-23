<?php

namespace Rubix\Engine\Metrics\Validation;

class Accuracy implements Classification
{
    /**
     * Test the accuracy of the predictions.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        $score = 0.0;

        foreach ($predictions as $i => $outcome) {
            if ($outcome === $labels[$i]) {
                $score++;
            }
        }

        return $score / (count($predictions) + self::EPSILON);
    }
}
