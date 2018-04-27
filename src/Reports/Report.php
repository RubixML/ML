<?php

namespace Rubix\Engine\Reports;

interface Report
{
    const EPSILON = 1e-8;

    /**
     * Score the predictions and display a report.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @return array
     */
    public function generate(array $predictions, array $outcomes) : array;
}
