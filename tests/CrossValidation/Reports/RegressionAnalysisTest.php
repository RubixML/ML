<?php

use Rubix\Engine\CrossValidation\Reports\Report;
use Rubix\Engine\CrossValidation\Reports\RegressionAnalysis;
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

        $labels = [11, 12, 14, 40, 55, 12, 16, 10, 2, 7];

        $result = [
            'mean_absolute_error' => 0.8000000100000001,
            'mean_squared_error' => 1.00000001,
            'rms_error' => 0.9999999995,
            'r_squared' => 0.9962367716957061,
        ];

        $report = $this->report->generate($predictions, $labels);

        $this->assertEquals($result, $report);
    }
}
