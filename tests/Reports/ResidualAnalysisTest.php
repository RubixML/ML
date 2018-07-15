<?php

namespace Rubix\Tests\Reports;

use Rubix\ML\Reports\Report;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Reports\ResidualAnalysis;
use PHPUnit\Framework\TestCase;

class ResidualAnalysisTest extends TestCase
{
    protected $report;

    protected $testing;

    protected $estimator;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], [], [], [], [], [], []],
            [11, 12, 14, 40, 55, 12, 16, 10, 2, 7]);

        $this->estimator = $this->createMock(Ridge::class);

        $this->estimator->method('predict')->willReturn([
            10, 12, 15, 42, 56, 12, 17, 9, 1, 7,
        ]);

        $this->report = new ResidualAnalysis();
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(ResidualAnalysis::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $actual = [
            'mean_absolute_error' => 0.8,
            'median_absolute_error' => 1.0,
            'mean_squared_error' => 1.0,
            'rms_error' => 1.0,
            'r_squared' => 0.9958930551562692,
            'min' => 0,
            'max' => 2,
            'variance' => 0.36000000000000004,
            'cardinality' => 10,
        ];

        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($actual, $result);
    }
}
