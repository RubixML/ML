<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\Classifier;

interface Classification extends Report
{
    /**
     * Generate the classification report.
     *
     * @param  \Rubix\ML\Classifiers\Classifier  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return array
     */
    public function generate(Classifier $estimator, Labeled $testing) : array;
}
