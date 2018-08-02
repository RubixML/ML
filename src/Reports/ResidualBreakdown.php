<?php

namespace Rubix\ML\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use MathPHP\Statistics\Average;
use Rubix\ML\Regressors\Regressor;
use MathPHP\Statistics\Descriptive;
use MathPHP\Statistics\RandomVariable;
use InvalidArgumentException;

/**
 * Residual Breakdown
 *
 * Residual Breakdown is a Report that measures the differences between the predicted
 * and actual values of a regression problem in detail. The statistics provided
 * in the report cover the first (mean), second (variance), third (skewness),
 * and fourth order (kurtosis) moments of the distribution of residuals produced
 * by a testing set.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ResidualBreakdown implements Report
{
    /**
     * Generate a residual analysis of a regression.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(Estimator $estimator, Dataset $testing) : array
    {
        if (!$estimator instanceof Regressor) {
            throw new InvalidArgumentException('This report only works on'
                . ' regressors.');
        }

        if (!$testing instanceof Labeled) {
            throw new InvalidArgumentException('This report requires a'
                . ' Labeled testing set.');
        }

        $errors = $l1 = $l2 = [];

        $sse = $sst = 0.0;

        $predictions = $estimator->predict($testing);

        $mean = Average::mean($testing->labels());

        foreach ($predictions as $i => $prediction) {
            $errors[] = $error = $testing->label($i) - $prediction;

            $l1[] = abs($error);
            $l2[] = $error ** 2;

            $sse += end($l2);
            $sst += ($testing->label($i) - $mean) ** 2;
        }

        $mse = Average::mean($l2);

        return [
            'mean_absolute_error' => Average::mean($l1),
            'median_absolute_error' => Average::median($l1),
            'mean_squared_error' => $mse,
            'rms_error' => sqrt($mse),
            'error_mean' => Average::mean($errors),
            'error_variance' => Descriptive::populationVariance($errors),
            'error_skewness' => RandomVariable::populationSkewness($errors),
            'error_kurtosis' => RandomVariable::kurtosis($errors),
            'error_min' => min($errors),
            'error_max' => max($errors),
            'r_squared' => 1 - ($sse / $sst),
            'cardinality' => $testing->numRows(),
        ];
    }
}
