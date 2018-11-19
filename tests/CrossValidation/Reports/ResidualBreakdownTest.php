<?php

namespace Rubix\ML\Tests\CrossValidation\Reports;

use Rubix\ML\CrossValidation\Reports\Report;
use Rubix\ML\CrossValidation\Reports\ResidualBreakdown;
use PHPUnit\Framework\TestCase;

class ResidualBreakdownTest extends TestCase
{
    protected $report;

    public function setUp()
    {
        $this->report = new ResidualBreakdown();
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(ResidualBreakdown::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $predictions = [10, 12, 15, 42, 56, 12, 17, 9, 1, 7,];

        $labels = [11, 12, 14, 40, 55, 12, 16, 10, 2, 7];

        $outcome = [
            'mean_absolute_error' => 0.8,
            'median_absolute_error' => 1,
            'mean_squared_error' => 1,
            'rms_error' => 1.0,
            'error_mean' => -0.2,
            'error_variance' => 0.9599999999999997,
            'error_skewness' => -0.22963966338592326,
            'error_kurtosis' => -1.0520833333333324,
            'error_min' => -2,
            'error_max' => 1,
            'r_squared' => 0.9958930551562692,
            'cardinality' => 10,
        ];

        $result = $this->report->generate($predictions, $labels);

        $this->assertEquals($outcome, $result);
    }
}
