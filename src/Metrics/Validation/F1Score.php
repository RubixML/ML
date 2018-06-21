<?php

namespace Rubix\ML\Metrics\Validation;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\Classifier;

class F1Score implements Classification
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return array
     */
    public function range() : array
    {
        return [0, 1];
    }
    
    /**
     * Score the average F1 score of the predictions.
     *
     * @param  \Rubix\ML\Classifiers\Classifier  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return float
     */
    public function score(Classifier $estimator, Labeled $testing) : float
    {
        $predictions = $estimator->predict($testing);

        $labels = $testing->labels();

        $classes = array_unique(array_merge($predictions, $labels));

        $truePositives = $falsePositives = $falseNegatives = [];

        foreach ($classes as $class) {
            $truePositives[$class] = $falsePositives[$class]
                = $falseNegatives[$class] = 0;
        }

        foreach ($predictions as $i => $outcome) {
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
