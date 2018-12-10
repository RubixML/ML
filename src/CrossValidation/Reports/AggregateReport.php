<?php

namespace Rubix\ML\CrossValidation\Reports;

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
                throw new InvalidArgumentException('Can only aggregate report'
                    . 'objects, ' . gettype($report) . ' found.');
            }
        }

        $this->reports = $reports;
    }

    /**
     * Generate the report.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return array
     */
    public function generate(array $predictions, array $labels) : array
    {
        $results = [];

        foreach ($this->reports as $index => $report) {
            $results[$index] = $report->generate($predictions, $labels);
        }

        return $results;
    }
}
