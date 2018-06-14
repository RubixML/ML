<?php

namespace Rubix\Engine\CrossValidation\Reports;

use MathPHP\Statistics\Average;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Regressors\Regressor;

class ResidualAnalysis implements Regression
{
    /**
     * Generate a residual analysis of a regression.
     *
     * @param  \Rubix\Engine\Regressors\Regressor  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return array
     */
    public function generate(Regressor $estimator, Labeled $testing) : array
    {
        $metrics = array_fill_keys(['error', 'sae', 'sse', 'sst'], 0.0);

        $mean = Average::mean($testing->labels());

        foreach ($estimator->predict($testing) as $i => $prediction) {
            $error = $testing->label($i) - $prediction;

            $metrics['sae'] += abs($error);
            $metrics['sse'] += $error ** 2;
            $metrics['sst'] += ($prediction - $mean) ** 2;
        }

        $n = $testing->numRows() + self::EPSILON;

        return [
            'mean_absolute_error' => $metrics['sae'] / $n,
            'mean_squared_error' => $metrics['sse'] / $n,
            'rms_error' => sqrt($metrics['sse'] / $n),
            'r_squared' => 1 - ($metrics['sse'] / ($metrics['sst'] + self::EPSILON)),
        ];
    }
}
