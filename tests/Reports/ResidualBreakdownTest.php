<?php

namespace Rubix\ML\Tests\Reports;

use Rubix\ML\Reports\Report;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Reports\ResidualBreakdown;
use PHPUnit\Framework\TestCase;

class ResidualBreakdownTest extends TestCase
{
    protected $report;

    protected $testing;

    protected $estimator;

    protected $outcome;

    public function setUp()
    {
        $samples = [[], [], [], [], [], [], [], [], [], []];

        $labels = [11, 12, 14, 40, 55, 12, 16, 10, 2, 7];

        $this->testing = Labeled::quick($samples, $labels);

        $this->estimator = $this->createMock(Ridge::class);

        $this->estimator->method('type')->willReturn(Ridge::REGRESSOR);

        $this->estimator->method('predict')->willReturn([
            10, 12, 15, 42, 56, 12, 17, 9, 1, 7,
        ]);

        $this->report = new ResidualBreakdown();
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(ResidualBreakdown::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
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

        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($outcome, $result);
    }
}
