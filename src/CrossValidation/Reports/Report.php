<?php

namespace Rubix\Engine\CrossValidation\Reports;

interface Report
{
    const EPSILON = 1e-8;

    /**
     * Generate the report.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return array
     */
    public function generate(array $predictions, array $labels) : array;
}
