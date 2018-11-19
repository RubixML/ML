<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;

/**
 * Residual Breakdown
 *
 * Residual Breakdown is a report that measures the differences between the predicted
 * and actual values of a regression problem in detail. The statistics provided
 * in the report cover the first (mean), second (variance), third (skewness),
 * and fourth (kurtosis) order moments of the distribution of residuals produced
 * by a testing set.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ResidualBreakdown implements Report
{
    /**
     * Generate the report.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(array $predictions, array $labels) : array
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        $muHat = Stats::mean($labels);

        $errors = $l1 = $l2 = [];

        $sse = $sst = 0.;

        foreach ($predictions as $i => $prediction) {
            $label = $labels[$i];

            $errors[] = $error = $label - $prediction;

            $l1[] = abs($error);
            $l2[] = $t = $error ** 2;

            $sse += $t;
            $sst += ($label - $muHat) ** 2;
        }

        list($mean, $variance) = Stats::meanVar($errors);

        $mse = Stats::mean($l2);

        $r2 = 1. - ($sse / ($sst ?: self::EPSILON));

        return [
            'mean_absolute_error' => Stats::mean($l1),
            'median_absolute_error' => Stats::median($l1),
            'mean_squared_error' => $mse,
            'rms_error' => sqrt($mse),
            'r_squared' => $r2,
            'error_mean' => $mean,
            'error_variance' => $variance,
            'error_skewness' => Stats::skewness($errors, $mean),
            'error_kurtosis' => Stats::kurtosis($errors, $mean),
            'error_min' => min($errors),
            'error_max' => max($errors),
            'cardinality' => count($predictions),
        ];
    }
}
