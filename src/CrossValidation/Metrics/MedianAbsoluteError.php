<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use MathPHP\Statistics\Average;
use Rubix\ML\Regressors\Regressor;
use InvalidArgumentException;

class MedianAbsoluteError implements Validation
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
     * Calculate the negative median absolute error of the predictions.
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

        $errors = [];

        foreach ($estimator->predict($testing) as $i => $prediction) {
            $errors[] = abs($testing->label($i) - $prediction);
        }

        return -Average::median($errors);
    }
}
