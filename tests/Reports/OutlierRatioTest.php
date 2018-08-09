<?php

namespace Rubix\Tests\Reports;

use Rubix\ML\Reports\Report;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Reports\OutlierRatio;
use Rubix\ML\AnomalyDetectors\RobustZScore;
use PHPUnit\Framework\TestCase;

class OutlierRatioTest extends TestCase
{
    protected $report;

    protected $estimator;

    protected $testing;

    protected $outcome;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], [], [], [], [], [], []],
            [11, 12, 14, 40, 55, 12, 16, 10, 2, 7]);

        $this->estimator = $this->createMock(RobustZScore::class);

        $this->estimator->method('predict')->willReturn([
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        ]);

        $this->report = new OutlierRatio();

        $this->outcome = [
            'outliers' => 1,
            'inliers' => 9,
            'ratio' => 0.11111111209876541,
            'cardinality' => 10,
        ];
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(OutlierRatio::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($this->outcome, $result);
    }
}
