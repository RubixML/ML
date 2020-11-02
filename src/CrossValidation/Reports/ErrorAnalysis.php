<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Report;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;

use function count;

use const Rubix\ML\EPSILON;

/**
 * Error Analysis
 *
 * The Error Analysis report measures the differences between the predicted and target values
 * of a regression problem using multiple error measurements (MAE, MSE, RMSE, MAPE, etc.) as
 * well as statistics regarding the distribution of errors.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ErrorAnalysis implements ReportGenerator
{
    /**
     * The estimator types that this report is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\EstimatorType>
     */
    public function compatibility() : array
    {
        return [
            EstimatorType::regressor(),
        ];
    }

    /**
     * Generate the report.
     *
     * @param list<int|float> $predictions
     * @param list<int|float> $labels
     * @return \Rubix\ML\Report
     */
    public function generate(array $predictions, array $labels) : Report
    {
        PredictionAndLabelCountsAreEqual::with($predictions, $labels)->check();

        $muHat = Stats::mean($labels);

        $errors = $l1 = $l2 = $are = $sle = [];

        $sse = $sst = 0.0;

        foreach ($predictions as $i => $prediction) {
            $label = $labels[$i];

            $errors[] = $error = $label - $prediction;

            $l1[] = abs($error);
            $l2[] = $se = $error ** 2;
            $are[] = abs($error / ($prediction ?: EPSILON));
            $sle[] = log((1.0 + $label) / ((1.0 + $prediction) ?: EPSILON)) ** 2;

            $sse += $se;
            $sst += ($label - $muHat) ** 2;
        }

        $mse = Stats::mean($l2);

        [$mean, $variance] = Stats::meanVar($errors);
        [$median, $mad] = Stats::medianMad($errors);

        $min = min($errors);
        $max = max($errors);

        return new Report([
            'mean_absolute_error' => Stats::mean($l1),
            'median_absolute_error' => Stats::median($l1),
            'mean_squared_error' => $mse,
            'mean_absolute_percentage_error' => 100.0 * Stats::mean($are),
            'rms_error' => sqrt($mse),
            'mean_squared_log_error' => Stats::mean($sle),
            'r_squared' => 1.0 - ($sse / ($sst ?: EPSILON)),
            'error_mean' => $mean,
            'error_midrange' => ($min + $max) / 2.0,
            'error_median' => $median,
            'error_variance' => $variance,
            'error_mad' => $mad,
            'error_iqr' => Stats::iqr($errors),
            'error_skewness' => Stats::skewness($errors, $mean),
            'error_kurtosis' => Stats::kurtosis($errors, $mean),
            'error_min' => $min,
            'error_max' => $max,
            'cardinality' => count($predictions),
        ]);
    }
}
