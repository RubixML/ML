<?php

namespace Rubix\Engine\Metrics\Reports;

use Rubix\Engine\Metrics\RMSError;
use Rubix\Engine\Metrics\RSquared;
use Rubix\Engine\Metrics\StandardError;
use Rubix\Engine\Metrics\MeanAbsoluteError;
use InvalidArgumentException;

class RegressionAnalysis implements Report
{
    /**
     * Generate a regression analysis.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @throws \InvalidArgumentException
     * @return void
     */
    public function generate(array $predictions, array $outcomes) : array
    {
        if (count($predictions) !== count($outcomes)) {
            throw new InvalidArgumentException('The number of outcomes must match the number of predictions.');
        }

        return [
            'mean_absolute_error' => (new MeanAbsoluteError())->score($predictions, $outcomes),
            'rms_error' => (new RMSError())->score($predictions, $outcomes),
            'r_squared' => (new RSquared())->score($predictions, $outcomes),
            'standard_error' => (new StandardError())->score($predictions, $outcomes),
        ];
    }
}
