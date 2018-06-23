<?php

namespace Rubix\ML\Metrics\Validation;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\Classifier;
use InvalidArgumentException;

class Accuracy implements Validation
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
     * Test the accuracy of the predictions.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Labeled  $testing
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(Estimator $estimator, Labeled $testing) : float
    {
        if (!$estimator instanceof Classifier) {
            throw new InvalidArgumentException('This metric only works on'
                . ' classifiers.');
        }

        $score = 0.0;

        foreach ($estimator->predict($testing) as $i => $prediction) {
            if ($prediction === $testing->label($i)) {
                $score++;
            }
        }

        return $score / ($testing->numRows() + self::EPSILON);
    }
}
