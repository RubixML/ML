<?php

namespace Rubix\Engine\Tests;

use MathPHP\Statistics\Average;
use InvalidArgumentException;

class RSquared extends Test
{
    /**
     * Calculate the coefficient of determination i.e. R^2 from the predictions.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @return float
     */
    public function score(array $predictions, array $outcomes) : float
    {
        $mean = Average::mean($predictions);
        $ssr = $sst = 0.0;

        foreach ($predictions as $i => $prediction) {
            $ssr += ($outcomes[$i] - $prediction) ** 2;
            $sst += ($outcomes[$i] - $mean) ** 2;
        }

        $r2 = 1 - ($ssr / $sst);

        $this->logger->log('Coefficient of determination (R^2): ' . number_format($r2, 5));

        return $r2;
    }
}
