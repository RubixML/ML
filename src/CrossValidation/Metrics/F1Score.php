<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

class F1Score implements Metric
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
        if ($estimator->type() !== Estimator::CLASSIFIER and $estimator->type() !== Estimator::DETECTOR) {
            throw new InvalidArgumentException('This metric only works with'
                . ' classifiers and anomaly detectors.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This metric requires a labeled'
                . ' testing set.');
        }

        if ($testing->numRows() === 0) {
            return 0.;
        }

        $predictions = $estimator->predict($testing);

        $labels = $testing->labels();

        $classes = array_unique(array_merge($predictions, $labels));

        $truePositives = $trueNegatives = $falsePositives = $falseNegatives =
            array_fill_keys($classes, 0);

        foreach ($predictions as $i => $outcome) {
            if ($outcome === $labels[$i]) {
                $truePositives[$outcome]++;
            } else {
                $falsePositives[$outcome]++;
                $falseNegatives[$labels[$i]]++;
            }
        }

        $score = 0.;

        foreach ($truePositives as $class => $tp) {
            $fp = $falsePositives[$class];
            $fn = $falseNegatives[$class];

            $score += ((2 * $tp) + self::EPSILON)
                / ((2 * $tp) + $fp + $fn + self::EPSILON);
        }

        return $score / count($classes);
    }
}
