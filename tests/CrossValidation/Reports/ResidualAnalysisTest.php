<?php

use Rubix\ML\Datasets\Labeled;
use Rubix\Tests\Helpers\MockRegressor;
use Rubix\ML\CrossValidation\Reports\Report;
use Rubix\ML\CrossValidation\Reports\Regression;
use Rubix\ML\CrossValidation\Reports\ResidualAnalysis;
use PHPUnit\Framework\TestCase;

class ResidualAnalysisTest extends TestCase
{
    protected $report;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], [], [], [], [], [], []],
            [11, 12, 14, 40, 55, 12, 16, 10, 2, 7]);

        $this->estimator = new MockRegressor([
            10, 12, 15, 42, 56, 12, 17, 9, 1, 7,
        ]);

        $this->report = new ResidualAnalysis();
    }

    public function test_build_regression_analysis()
    {
        $this->assertInstanceOf(ResidualAnalysis::class, $this->report);
        $this->assertInstanceOf(Regression::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $actual = [
            'mean_absolute_error' => 0.7999999992,
            'mean_squared_error' => 0.9999999989999999,
            'rms_error' => 0.9999999995,
            'r_squared' => 0.9962367816957203,
        ];

        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($actual, $result);
    }
}
