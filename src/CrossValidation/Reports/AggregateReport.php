<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Report;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

/**
 * Aggregate Report
 *
 * A report generator that aggregates the output of multiple reports.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AggregateReport implements ReportGenerator
{
    /**
     * The report middleware stack. i.e. the reports to generate when the reports
     * method is called.
     *
     * @var \Rubix\ML\CrossValidation\Reports\ReportGenerator[]
     */
    protected $reports = [
        //
    ];

    /**
     * The estimator compatibility of the aggregate.
     *
     * @var \Rubix\ML\EstimatorType[]
     */
    protected $compatibility;

    /**
     * @param \Rubix\ML\CrossValidation\Reports\ReportGenerator[] $reports
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(array $reports)
    {
        if (empty($reports)) {
            throw new InvalidArgumentException('Aggregate must contain'
                . ' at least 1 sub report.');
        }

        $compatibilities = [];

        foreach ($reports as $report) {
            if (!$report instanceof ReportGenerator) {
                throw new InvalidArgumentException('Sub report must'
                    . ' implement the ReportGenerator interface.');
            }

            $compatibilities[] = $report->compatibility();
        }

        $compatibility = array_intersect(...$compatibilities);

        if (count($compatibility) < 1) {
            throw new InvalidArgumentException('Report must contain'
                . ' sub reports that have at least 1 compatible'
                . ' Estimator type in common.');
        }

        $this->reports = $reports;
        $this->compatibility = array_values($compatibility);
    }

    /**
     * The estimator types that this report is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\EstimatorType>
     */
    public function compatibility() : array
    {
        return $this->compatibility;
    }

    /**
     * Generate the report.
     *
     * @param list<string|int|float> $predictions
     * @param list<string|int|float> $labels
     * @return \Rubix\ML\Report
     */
    public function generate(array $predictions, array $labels) : Report
    {
        $report = [];

        foreach ($this->reports as $name => $subReport) {
            $report[$name] = $subReport->generate($predictions, $labels);
        }

        return new Report($report);
    }
}
