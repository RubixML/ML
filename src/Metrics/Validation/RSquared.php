<?php

namespace Rubix\ML\Metrics\Validation;

use MathPHP\Statistics\Average;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Regressor;

class RSquared implements Regression
{
    /**
     * Calculate the coefficient of determination i.e. R^2 from the predictions.
     *
     * @param  \Rubix\ML\Regressors\Regressor  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return float
     */
    public function score(Regressor $estimator, Labeled $testing) : float
    {
        $mean = Average::mean($testing->labels());

        $ssr = $sst = 0.0;

        foreach ($estimator->predict($testing) as $i => $prediction) {
            $ssr += ($testing->label($i) - $prediction) ** 2;
            $sst += ($testing->label($i) - $mean) ** 2;
        }

        return 1 - ($ssr / ($sst + self::EPSILON));
    }
}
