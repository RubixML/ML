<?php

namespace Rubix\Engine;

use League\CLImate\CLImate;
use Rubix\Engine\Reports\Report;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Persisters\Persistable;
use ReflectionClass;

class Prototype implements Estimator, Persistable
{
    /**
     * The base estimator.
     *
     * @var \Rubix\Engine\Estimator
     */
    protected $estimator;

    /**
     * The report middleware stack. i.e. the reports to generate when the reports
     * method is called.
     *
     * @var array
     */
    protected $reports = [
        //
    ];

    /**
     * @param  \Rubix\Engine\Estimator  $estimator
     * @param  array  $reports
     * @return void
     */
    public function __construct(Estimator $estimator, array $reports = [])
    {
        foreach ($reports as $report) {
            $this->addReport($report);
        }

        $this->estimator = $estimator;
    }

    /**
     * Return the underlying estimator instance.
     *
     * @return \Rubix\Engine\Estimator
     */
    public function estimator() : Estimator
    {
        return $this->estimator;
    }

    /**
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        $this->estimator->train($dataset);
    }

    /**
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        return $this->estimator->predict($sample);
    }

    /**
     * Generate the reports in the reporting middleware stack.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @return void
     */
    public function test(Supervised $dataset) : void
    {
        $logger = new CLImate();

        $predictions = array_map(function ($sample) {
            return $this->predict($sample)->outcome();
        }, $dataset->samples());

        foreach ($this->reports as $report) {
            $logger->json($report->generate($predictions, $dataset->outcomes()));
        }
    }

    /**
     * Add a test to the testing stack.
     *
     * @param  \Rubix\Engine\Metrics\Reports\Report  $report
     * @return self
     */
    public function addReport(Report $report) : self
    {
        $this->reports[] = $report;

        return $this;
    }
}
