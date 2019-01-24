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
     * The estimator compatibility of the aggregate.
     * 
     * @var int[]
     */
    protected $compatibility;

    /**
     * @param  array  $reports
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $reports)
    {
        if (empty($reports)) {
            throw new InvalidArgumentException('Cannot generate an aggregate'
                . ' of less than 1 report.');
        }

        $compatibilities = [];

        foreach ($reports as $report) {
            if (!$report instanceof Report) {
                throw new InvalidArgumentException('Can only aggregate report'
                    . 'objects, ' . gettype($report) . ' found.');
            }

            $compatibilities[] = $report->compatibility();
        }

        $compatibility = array_intersect(...$compatibilities);

        if (count($compatibility) < 1) {
            throw new InvalidArgumentException('Aggregate must only contain'
                . ' reports that share at least 1 estimator they are'
                . ' compatible in common with.');
        }

        $this->reports = $reports;
        $this->compatibility = array_values($compatibility);
    }

    /**
     * The estimator types that this report is compatible with.
     * 
     * @return int[]
     */
    public function compatibility() : array
    {
        return $this->compatibility;
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
