<?php

namespace Rubix\ML\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;

/**
 * Prediction Speed
 *
 * This Report measures the prediction speed of an Estimator given as the number
 * of predictions per second (PPM) as well as the average time to make a single
 * prediction.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PredictionSpeed implements Report
{
    /**
     * Generate the report.
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
            'ppm' => ($testing->numRows() / ($total + self::EPSILON)) * 60,
            'average_seconds' => $total / ($testing->numRows() + self::EPSILON),
            'total_seconds' => $total,
            'cardinality' => $testing->numRows(),
        ];
    }
}
