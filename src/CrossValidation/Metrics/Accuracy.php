<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

/**
 * Accuracy
 *
 * A simple metric that measures the true positive rate over the entire testing
 * set.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Accuracy implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [0., 1.];
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
        if ($estimator->type() !== Estimator::CLASSIFIER and $estimator->type() !== Estimator::DETECTOR) {
            throw new InvalidArgumentException('This metric only works with'
                . ' classifiers and anomaly detectors.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This metric requires a labeled'
                . ' testing set.');
        }

        $n = $testing->numRows();

        if ($n < 1) {
            return 0.;
        }

        $score = 0.;

        foreach ($estimator->predict($testing) as $i => $prediction) {
            if ($prediction === $testing->label($i)) {
                $score++;
            }
        }

        return $score / $n;
    }
}
