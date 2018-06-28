<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;

class PredictionSpeed implements Report
{
    /**
     * Generate a confusion matrix.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @return array
     */
    public function generate(Estimator $estimator, Dataset $testing) : array
    {
        $start = microtime(true);

        $estimator->predict($testing);

        $end = microtime(true);

        $total = $end - $start;

        return [
            'pps' => $testing->numRows() / ($total + self::EPSILON),
            'average_sec' => $total / ($testing->numRows() + self::EPSILON),
            'total_sec' => $total,
            'cardinality' => $testing->numRows(),
        ];
    }
}
