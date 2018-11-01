<?php

namespace Rubix\ML\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

/**
 * Outlier Ratio
 *
 * Outlier Ratio is the ratio of outliers to inliers in an Anomaly Detection problem.
 * It can be used to gauge the amount of contamination that a detector is predicting.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class OutlierRatio implements Report
{
    /**
     * Generate the report.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(Estimator $estimator, Dataset $testing) : array
    {
        if ($estimator->type() !== Estimator::DETECTOR) {
            throw new InvalidArgumentException('This report only works with'
                . ' detectors.');
        }

        $n = $testing->numRows();

        if ($n < 1) {
            throw new InvalidArgumentException('Testing set must contain at'
                . ' least one sample to predict.');
        }

        $counts = array_count_values($estimator->predict($testing));

        $outliers = $counts[1] ?? 0;
        $inliers = $counts[0] ?? 0;

        $ratio = $outliers / ($inliers ?: self::EPSILON);

        $proportion = $outliers / $n;

        return [
            'ratio' => $ratio,
            'proportion' => $proportion,
            'percentage' => 100. * $proportion,
            'outliers' => $outliers,
            'inliers' => $inliers,
            'cardinality' => $n,
        ];
    }
}
