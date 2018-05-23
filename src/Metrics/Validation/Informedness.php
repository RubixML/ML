<?php

namespace Rubix\Engine\Metrics\Validation;

class Informedness implements Classification
{
    /**
     * Calculate the informedness score of the predicted classes. Informedness
     * is determined by recall + specificity - 1.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        $classes = array_unique(array_merge($predictions, $labels));

        $truePositives = $trueNegatives = $falsePositives
            = $falseNegatives = [];

        foreach ($classes as $class) {
            $truePositives[$class] = $trueNegatives[$class]
                = $falsePositives[$class] = $falseNegatives[$class] = 0;
        }

        foreach ($predictions as $i => $outcome) {
            if ($outcome === $labels[$i]) {
                $truePositives[$outcome]++;

                foreach ($classes as $class) {
                    if ($class !== $outcome) {
                        $trueNegatives[$class]++;
                    }
                }
            } else {
                $falsePositives[$outcome]++;
                $falseNegatives[$labels[$i]]++;
            }
        }

        $score = 0.0;

        foreach ($truePositives as $class => $tp) {
            $tn = $trueNegatives[$class];
            $fp = $falsePositives[$class];
            $fn = $falseNegatives[$class];

            $score += $tp / ($tp + $fn + self::EPSILON)
                + $tn / ($tn + $fp + self::EPSILON)
                - 1;
        }

        return $score / (count($classes) + self::EPSILON);
    }
}
