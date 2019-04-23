<?php

namespace Rubix\ML\CrossValidation\Reports;

interface Report
{
    /**
     * The estimator types that this report is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array;

    /**
     * Generate the report.
     *
     * @param array $predictions
     * @param array $labels
     * @return array
     */
    public function generate(array $predictions, array $labels) : array;
}
