<?php

namespace Rubix\Engine\Metrics;

use MathPHP\Statistics\Correlation;
use MathPHP\Statistics\Average;
use InvalidArgumentException;

class RSquared implements Regression
{
    /**
     * Calculate the coefficient of determination i.e. R^2 from the predictions.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $outcomes) : float
    {
        if (count($predictions) !== count($outcomes)) {
            throw new InvalidArgumentException('The number of outcomes must match the number of predictions.');
        }

        $mean = Average::mean($outcomes);
        $ssr = $sst = 0.0;

        foreach ($predictions as $i => $prediction) {
            $ssr += ($outcomes[$i] - $prediction) ** 2;
            $sst += ($outcomes[$i] - $mean) ** 2;
        }

        return 1 - ($ssr / $sst);
    }

    /**
     * Should this metric be minimized?
     *
     * @return bool
     */
    public function minimize() : bool
    {
        return false;
    }
}
