<?php

namespace Rubix\ML\CrossValidation\Reports;

use InvalidArgumentException;

use function count;
use function gettype;

/**
 * Aggregate Report
 *
 * A report that aggregates the output of multiple reports into one report.
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
     * @var \Rubix\ML\CrossValidation\Reports\Report[]
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
     * @param \Rubix\ML\CrossValidation\Reports\Report[] $reports
     * @throws \InvalidArgumentException
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
     * @param mixed[] $predictions
     * @param mixed[] $labels
     * @return mixed[]
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
