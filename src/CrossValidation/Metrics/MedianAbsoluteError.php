<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;

/**
 * Median Absolute Error
 *
 * Median Absolute Error (MAE) is a robust measure of the error that ignores
 * highly erroneous predections.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MedianAbsoluteError implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-INF, 0.];
    }

    /**
     * Calculate the negative median absolute error of the predictions.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(Estimator $estimator, Dataset $testing) : float
    {
        if ($estimator->type() !== Estimator::REGRESSOR) {
            throw new InvalidArgumentException('This metric only works with'
                . ' regresors.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This metric requires a labeled'
                . ' testing set.');
        }

        if ($testing->numRows() === 0) {
            return 0.;
        }

        $errors = [];

        foreach ($estimator->predict($testing) as $i => $prediction) {
            $errors[] = abs($testing->label($i) - $prediction);
        }

        return -Stats::median($errors);
    }
}
