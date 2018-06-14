<?php

namespace Rubix\Engine\Metrics\Validation;

use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Regressors\Regressor;

class MeanSquaredError implements Regression
{
    /**
     * Calculate the negative mean squared error of the predictions.
     *
     * @param  \Rubix\Engine\Regressors\Regressor  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return float
     */
    public function score(Regressor $estimator, Labeled $testing) : float
    {
        $error = 0.0;

        foreach ($estimator->predict($testing) as $i => $prediction) {
            $error += ($testing->label($i) - $prediction) ** 2;
        }

        return -($error / ($testing->numRows() + self::EPSILON));
    }
}
