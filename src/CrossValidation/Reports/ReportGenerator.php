<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\CrossValidation\Reports\Results\Report;

interface ReportGenerator
{
    /**
     * The estimator types that this report is compatible with.
     *
     * @return \Rubix\ML\EstimatorType[]
     */
    public function compatibility() : array;

    /**
     * Generate the report.
     *
     * @param (string|int|float)[] $predictions
     * @param (string|int|float)[] $labels
     * @return \Rubix\ML\CrossValidation\Reports\Results\Report
     */
    public function generate(array $predictions, array $labels) : Report;
}
