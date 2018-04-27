<?php

namespace Rubix\Engine\Reports;

use MathPHP\Statistics\Average;
use InvalidArgumentException;

class RegressionAnalysis implements Report
{
    /**
     * Generate a regression analysis.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @throws \InvalidArgumentException
     * @return void
     */
    public function generate(array $predictions, array $outcomes) : array
    {
        if (count($predictions) !== count($outcomes)) {
            throw new InvalidArgumentException('The number of outcomes must match the number of predictions.');
        }

        $metrics = array_fill_keys(['error', 'sae', 'sse', 'sst'], 0.0);

        $mean = Average::mean($outcomes);
        $n = count($outcomes);

        foreach ($predictions as $i => $prediction) {
            $error = $outcomes[$i] - $prediction;
            $metrics['sae'] += abs($error);
            $metrics['sse'] += $error ** 2;
            $metrics['sst'] += ($outcomes[$i] - $mean) ** 2;
        }

        return [
            'mean_absolute_error' => $metrics['sae'] / $n,
            'mean_squared_error' => $metrics['sse'] / $n,
            'rms_error' => sqrt($metrics['sse'] / $n),
            'r_squared' => 1 - ($metrics['sse'] / $metrics['sst']),
        ];
    }
}
