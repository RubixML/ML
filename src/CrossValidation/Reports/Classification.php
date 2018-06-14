<?php

namespace Rubix\Engine\CrossValidation\Reports;

use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Classifiers\Classifier;

interface Classification extends Report
{
    /**
     * Generate the classification report.
     *
     * @param  \Rubix\Engine\Classifiers\Classifier  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return array
     */
    public function generate(Classifier $estimator, Labeled $testing) : array;
}
