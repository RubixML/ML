<?php

namespace Rubix\Engine\Metrics\Validation;

class F1Score implements Classification
{
    /**
     * Score the average F1 score of the predictions.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        $outcomes = [];

        foreach ($predictions as $prediction) {
            $outcomes[] = $prediction->outcome();
        }

        $classes = array_unique(array_merge($outcomes, $labels));

        $truePositives = $falsePositives = $falseNegatives = [];

        foreach ($classes as $class) {
            $truePositives[$class] = $falsePositives[$class]
                = $falseNegatives[$class] = 0;
        }

        foreach ($outcomes as $i => $outcome) {
            if ($outcome === $labels[$i]) {
                $truePositives[$outcome]++;
            } else {
                $falsePositives[$outcome]++;
                $falseNegatives[$labels[$i]]++;
            }
        }

        $score = 0.0;

        foreach ($truePositives as $class => $tp) {
            $fp = $falsePositives[$class];
            $fn = $falseNegatives[$class];

            $score += ((2 * $tp) / ((2 * $tp) + $fp + $fn) + self::EPSILON);
        }

        return $score / (count($classes) + self::EPSILON);
    }
}
