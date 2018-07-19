<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\Classifier;
use Rubix\ML\AnomalyDetectors\Detector;
use InvalidArgumentException;

class F1Score implements Validation
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

        if ($testing->numRows() === 0) {
            return 0.0;
        }

        return $score / count($classes);
    }
}
