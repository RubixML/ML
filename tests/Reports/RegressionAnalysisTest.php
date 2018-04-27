<?php

use Rubix\Engine\Reports\Report;
use Rubix\Engine\Reports\RegressionAnalysis;
use PHPUnit\Framework\TestCase;

class RegressionAnalysisTest extends TestCase
{
    protected $report;

    public function setUp()
    {
        $this->report = new RegressionAnalysis();
    }

    public function test_build_regression_analysis()
    {
        $this->assertInstanceOf(RegressionAnalysis::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $predictions = [10, 12, 15, 42, 56, 12, 17, 9, 1, 7];
        $outcomes = [11, 12, 14, 40, 55, 12, 16, 10, 2, 7];

        $result = [
            'mean_absolute_error' => 0.8,
            'mean_squared_error' => 1.0,
            'rms_error' => 1.0,
            'r_squared' => 0.9958930551562692,
        ];

        $this->assertEquals($result, $this->report->generate($predictions, $outcomes));
    }
}
