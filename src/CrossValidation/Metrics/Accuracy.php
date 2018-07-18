<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\Classifier;
use Rubix\ML\AnomalyDetectors\Detector;
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
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(Estimator $estimator, Dataset $testing) : float
    {
        if (!$estimator instanceof Classifier and !$estimator instanceof Detector) {
            throw new InvalidArgumentException('This metric only works on'
                . ' classifiers and anomaly detectors.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This metric requires a labeled'
                . ' testing set.');
        }

        $score = 0.0;

        if ($testing->numRows() === 0) {
            return $score;
        }

        foreach ($estimator->predict($testing) as $i => $prediction) {
            if ($prediction === $testing->label($i)) {
                $score++;
            }
        }

        return $score / $testing->numRows();
    }
}
