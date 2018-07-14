<?php

namespace Rubix\ML\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;

/**
 * Aggregate Report
 *
 * A Report that aggregates the results of multiple reports. The reports are
 * indexed by their order given at construction time.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AggregateReport implements Report
{
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
     * @param  array  $reports
     * @return void
     */
    public function __construct(array $reports)
    {
        foreach ($reports as $index => $report) {
            $this->addReport($report, $index);
        }
    }

    /**
     * Generate an aggregated report consisting of 1 or more individual reports.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @return array
     */
    public function generate(Estimator $estimator, Dataset $testing) : array
    {
        $reports = [];

        foreach ($this->reports as $index => $report) {
            $reports[$index] = $report->generate($estimator, clone $testing);
        }

        return $reports;
    }

    /**
     * Add a report to the stack.
     *
     * @param  \Rubix\ML\Reports\Report  $report
     * @param  mixed  $index
     * @return void
     */
    public function addReport(Report $report, $index) : void
    {
        $this->reports[$index] = $report;
    }
}
