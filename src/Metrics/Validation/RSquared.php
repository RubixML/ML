<?php

namespace Rubix\Engine\Metrics\Validation;

use MathPHP\Statistics\Average;

class RSquared implements Regression
{
    /**
     * Calculate the coefficient of determination i.e. R^2 from the predictions.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        $mean = Average::mean($labels);
        $ssr = $sst = 0.0;

        foreach ($predictions as $i => $prediction) {
            $ssr += ($labels[$i] - $prediction->outcome()) ** 2;
            $sst += ($labels[$i] - $mean) ** 2;
        }

        return 1 - ($ssr / ($sst + self::EPSILON));
    }
}
