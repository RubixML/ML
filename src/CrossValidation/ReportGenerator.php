<?php

namespace Rubix\Engine\CrossValidation;

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Estimator;
use Rubix\Engine\CrossValidation\Reports\Report;
use InvalidArgumentException;

class ReportGenerator
{
    /**
     * The report to generate.
     *
     * @var \Rubix\Engine\CrossValidation\Reports\Report
     */
    protected $report;

    /**
     * The holdout ratio. i.e. the ratio of samples to use for validation.
     *
     * @var float
     */
    protected $ratio;

    /**
     * @param  \Rubix\Engine\CrossValidation\Reports\Report  $report
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(Report $report, float $ratio = 0.2)
    {
        if ($ratio < 0.01 || $ratio > 1.0) {
            throw new InvalidArgumentException('Holdout ratio must be'
                . ' between 0.01 and 1.0.');
        }

        $this->report = $report;
        $this->ratio = $ratio;
    }

    /**
     * Generate the report.
     *
     * @param  \Rubix\Engine\Estimators\Estimator  $estimator
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @return array
     */
    public function generate(Estimator $estimator, Supervised $dataset) : array
    {
        if ($estimator instanceof Classifier) {
            list($training, $testing) = $dataset->randomize()
                ->stratifiedSplit(1 - $this->ratio);
        } else {
            list($training, $testing) = $dataset->randomize()
                ->split(1 - $this->ratio);
        }

        $estimator->train($training);

        $predictions = $estimator->predict($testing);

        $report = $this->report->generate($predictions, $testing->labels());

        return $report;
    }
}
