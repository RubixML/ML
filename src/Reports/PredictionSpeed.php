<?php

namespace Rubix\ML\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

/**
 * Prediction Speed
 *
 * This Report measures the prediction speed of an Estimator given as the number
 * of predictions per second (PPS) as well as the average time to make a single
 * prediction.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PredictionSpeed implements Report
{
    const MIN_TEST_SIZE = 200;

    /**
     * The ratio of test samples to use to generate the report.
     * 
     * @var float
     */
    protected $ratio;

    /**
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $ratio = 0.2)
    {
        if ($ratio < 0. or $ratio > 1.) {
            throw new InvalidArgumentException("Ratio must be between"
                . " 0 and 1, $ratio given.");
        }

        $this->ratio = $ratio;
    }

    /**
     * Generate the report.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @return array
     */
    public function generate(Estimator $estimator, Dataset $testing) : array
    {
        $n = $testing->numRows();

        if ($n < 1) {
            throw new InvalidArgumentException('Testing set must contain at'
                . ' least one sample to predict.');
        }

        $p = (int) round(max(self::MIN_TEST_SIZE, $this->ratio * $n));

        $subset = $testing->randomize()->head($p);

        $start = microtime(true);

        $estimator->predict($subset);

        $end = microtime(true);

        $runtime = $end - $start;

        $pps = $n / ($runtime  ?: self::EPSILON);

        $average = $runtime / $n;

        return [
            'pps' => $pps,
            'ppm' => 60. * $pps,
            'total_seconds' => $runtime,
            'average_seconds' => $average,
            'cardinality' => $n,
        ];
    }
}
