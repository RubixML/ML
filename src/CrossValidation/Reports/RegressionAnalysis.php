<?php

namespace Rubix\Engine\CrossValidation\Reports;

use MathPHP\Statistics\Average;

class RegressionAnalysis implements Report
{
    /**
     * Generate a regression analysis.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return array
     */
    public function generate(array $predictions, array $labels) : array
    {
        $metrics = array_fill_keys(['error', 'sae', 'sse', 'sst'], 0.0);

        $mean = Average::mean($labels);

        foreach ($predictions as $i => $prediction) {
            $error = $labels[$i] - $prediction->outcome();

            $metrics['sae'] += abs($error);
            $metrics['sse'] += $error ** 2;
            $metrics['sst'] += ($prediction->outcome() - $mean) ** 2;
        }

        $n = count($labels);

        return [
            'mean_absolute_error' => $metrics['sae'] / $n + self::EPSILON,
            'mean_squared_error' => $metrics['sse'] / $n + self::EPSILON,
            'rms_error' => sqrt($metrics['sse'] / ($n + self::EPSILON)),
            'r_squared' => 1 - ($metrics['sse'] / $metrics['sst'] + self::EPSILON),
        ];
    }
}
