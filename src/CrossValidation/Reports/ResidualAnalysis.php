<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use MathPHP\Statistics\Average;
use Rubix\ML\Regressors\Regressor;
use InvalidArgumentException;

class ResidualAnalysis implements Report
{
    /**
     * Generate a residual analysis of a regression.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Runix\ML\Datasets\Labeled  $testing
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(Estimator $estimator, Labeled $testing) : array
    {
        if (!$estimator instanceof Regressor) {
            throw new InvalidArgumentException('This report only works on'
                . ' regressors.');
        }

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
