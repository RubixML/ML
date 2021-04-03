<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Report;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
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

        return new Report([
            'mean absolute error' => Stats::mean($l1),
            'median absolute error' => Stats::median($l1),
            'mean squared error' => $mse,
            'mean squared log error' => Stats::mean($sle),
            'mean absolute percentage error' => 100.0 * Stats::mean($are),
            'rms error' => sqrt($mse),
            'r squared' => 1.0 - ($sse / ($sst ?: EPSILON)),
            'error mean' => $mean,
            'error median' => $median,
            'error variance' => $variance,
            'error stddev' => sqrt($variance),
            'error mad' => $mad,
            'error iqr' => Stats::iqr($errors),
            'error skewness' => Stats::skewness($errors, $mean),
            'error kurtosis' => Stats::kurtosis($errors, $mean),
            'error min' => min($errors),
            'error max' => max($errors),
            'cardinality' => count($predictions),
        ]);
    }
}
