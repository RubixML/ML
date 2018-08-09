<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use MathPHP\Statistics\Average;
use Rubix\ML\Regressors\Regressor;
use InvalidArgumentException;

class RSquared implements Validation
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return array
     */
    public function range() : array
    {
        return [-INF, 1];
    }

    /**
     * Calculate the coefficient of determination i.e. R^2 from the predictions.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(Estimator $estimator, Dataset $testing) : float
    {
        if (!$estimator instanceof Regressor) {
            throw new InvalidArgumentException('This metric only works on'
                . ' regresors.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This metric requires a labeled'
                . ' testing set.');
        }

        if ($testing->numRows() === 0) {
            return 0.0;
        }

        $mean = Average::mean($testing->labels());

        $ssr = $sst = 0.0;

        foreach ($estimator->predict($testing) as $i => $prediction) {
            $ssr += ($testing->label($i) - $prediction) ** 2;
            $sst += ($testing->label($i) - $mean) ** 2;
        }

        return 1 - (($ssr + self::EPSILON) / ($sst + self::EPSILON));
    }
}
