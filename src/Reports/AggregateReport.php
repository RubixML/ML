<?php

namespace Rubix\ML\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

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
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $reports)
    {
        if (empty($reports)) {
            throw new InvalidArgumentException('Cannot generate an aggregate of'
                . ' less than 1 report.');
        }

        foreach ($reports as $index => $report) {
            if (!$report instanceof Report) {
                throw new InvalidArgumentException('Can only aggregate reports'
                    . ', ' . gettype($report) . ' found.');
            }

            $this->reports[$index] = $report;
        }
    }

    /**
     * Generate the report.
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
}
