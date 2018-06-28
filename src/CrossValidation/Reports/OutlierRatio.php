<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\AnomalyDetectors\Detector;
use InvalidArgumentException;

class OutlierRatio implements Report
{
    /**
     * Generate a confusion matrix.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(Estimator $estimator, Dataset $testing) : array
    {
        if (!$estimator instanceof Detector) {
            throw new InvalidArgumentException('This report only works on'
                . ' anomaly detectors.');
        }

        $counts = array_count_values($estimator->predict($testing));

        return [
            'outliers' => $counts[1],
            'inliers' => $counts[0],
            'ratio' => $counts[1] / ($counts[0] + self::EPSILON),
            'cardinality' => $testing->numRows(),
        ];
    }
}
