<?php

namespace Rubix\ML\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

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
        if ($testing->numRows() === 0) {
            throw new InvalidArgumentException('Testing set must contain at'
                . ' least one sample.');
        }

        $start = microtime(true);

        $estimator->predict($testing);

        $end = microtime(true);

        $total = $end - $start;

        $pps = ($testing->numRows() + self::EPSILON) / ($total + self::EPSILON);

        return [
            'pps' => $pps,
            'ppm' => 60 * $pps,
            'average_seconds' => $total / ($testing->numRows() + self::EPSILON),
            'total_seconds' => $total,
            'cardinality' => $testing->numRows(),
        ];
    }
}
