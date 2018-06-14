<?php

namespace Rubix\ML\Metrics\Validation;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\Classifier;

class Accuracy implements Classification
{
    /**
     * Test the accuracy of the predictions.
     *
     * @param  \Rubix\ML\Classifiers\Classifier  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return float
     */
    public function score(Classifier $estimator, Labeled $testing) : float
    {
        $score = 0.0;

        foreach ($estimator->predict($testing) as $i => $prediction) {
            if ($prediction === $testing->label($i)) {
                $score++;
            }
        }

        return $score / ($testing->numRows() + self::EPSILON);
    }
}
