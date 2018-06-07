<?php

namespace Rubix\Engine\CrossValidation;

use Rubix\Engine\Estimator;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Clusterers\Clusterer;
use Rubix\Engine\Classifiers\Classifier;
use Rubix\Engine\CrossValidation\Reports\Report;
use Rubix\Engine\CrossValidation\Reports\Regression;
use Rubix\Engine\CrossValidation\Reports\Clustering;
use Rubix\Engine\CrossValidation\Reports\Classification;
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
        if ($ratio < 0.01 or $ratio > 1.0) {
            throw new InvalidArgumentException('Holdout ratio must be'
                . ' between 0.01 and 1.0.');
        }

        $this->report = $report;
        $this->ratio = $ratio;
    }

    /**
     * Generate the report.
     *
     * @param  \Rubix\Engine\Estimator  $estimator
     * @param  \Rubix\Engine\Datasets\Labeled  $dataset
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(Estimator $estimator, Labeled $dataset) : array
    {
        if ($estimator instanceof Classifier) {
            if (!$this->report instanceof Classification) {
                throw new InvalidArgumentException('Classification report only'
                    . ' works on Classifiers, ' . get_class($estimator)
                    . ' found.');
            }
        }

        if ($estimator instanceof Regressor) {
            if (!$this->report instanceof Regression) {
                throw new InvalidArgumentException('Regression report only'
                    . ' works on Regressors, ' . get_class($estimator)
                    . ' found.');
            }
        }

        if ($estimator instanceof Clusterer) {
            if (!$this->report instanceof Clustering) {
                throw new InvalidArgumentException('Clustering report only'
                    . ' works on Clusterers, ' . get_class($estimator)
                    . ' found.');
            }
        }

        if ($estimator instanceof Classifier or $estimator instanceof Clusterer) {
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
