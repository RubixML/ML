<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

/**
 * Root Mean Squared Error
 *
 * Root Mean Squared Error or average L2 loss is a metric that is used to
 * measure the residuals of a regression problem.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RMSError implements Metric
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
     * Calculate the negative root mean square error from the predictions.
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

        $n = $testing->numRows();

        if ($n < 1) {
            return 0.;
        }

        $error = 0.;

        foreach ($estimator->predict($testing) as $i => $prediction) {
            $error += ($testing->label($i) - $prediction) ** 2;
        }

        return -sqrt($error / $n);
    }
}
