<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;

use const Rubix\ML\EPSILON;

/**
 * Residual Analysis
 *
 * Residual Analysis is a report that measures the differences between the predicted
 * and actual values of a regression problem in detail.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ResidualAnalysis implements Report
{
    /**
     * The estimator types that this report is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            Estimator::REGRESSOR,
        ];
    }

    /**
     * Generate the report.
     *
     * @param array $predictions
     * @param array $labels
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

        $errors = $l1 = $l2 = $ape = $spe = $log = [];

        $sse = $sst = 0.;

        foreach ($predictions as $i => $prediction) {
            $label = $labels[$i];

            $errors[] = $error = $label - $prediction;

            $l1[] = abs($error);
            $l2[] = $se = $error ** 2;
            $ape[] = abs($error / ($prediction ?: EPSILON)) * 100.;
            $log[] = log((1. + $label) / ((1. + $prediction) ?: EPSILON)) ** 2;

            $sse += $se;
            $sst += ($label - $muHat) ** 2;
        }

        [$mean, $variance] = Stats::meanVar($errors);

        $mse = Stats::mean($l2);

        return [
            'mean_absolute_error' => Stats::mean($l1),
            'median_absolute_error' => Stats::median($l1),
            'mean_absolute_percentage_error' => Stats::mean($ape),
            'mean_squared_error' => $mse,
            'rms_error' => sqrt($mse),
            'mean_squared_log_error' => Stats::mean($log),
            'r_squared' => 1. - ($sse / ($sst ?: EPSILON)),
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
