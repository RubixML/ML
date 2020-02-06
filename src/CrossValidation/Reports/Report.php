<?php

namespace Rubix\ML\CrossValidation\Reports;

interface Report
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
     * @return mixed[]
     */
    public function generate(array $predictions, array $labels) : array;
}
