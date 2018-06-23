<?php

namespace Rubix\ML\Metrics\Validation;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Regressor;
use InvalidArgumentException;

class MeanAbsoluteError implements Validation
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return array
     */
    public function range() : array
    {
        return [-INF, 0];
    }

    /**
     * Calculate the negative mean absolute error of the predictions.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Labeled  $testing
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(Estimator $estimator, Labeled $testing) : float
    {
        if (!$estimator instanceof Regressor) {
            throw new InvalidArgumentException('This metric only works on'
                . ' regresors.');
        }

        $error = 0.0;

        foreach ($estimator->predict($testing) as $i => $prediction) {
            $error += abs($testing->label($i) - $prediction);
        }

        return -($error / ($testing->numRows() + self::EPSILON));
    }
}
