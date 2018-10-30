<?php

namespace Rubix\ML\Tests\Reports;

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

    public function setUp()
    {
        $samples = [[], [], [], [], [], [], [], [], [], []];

        $labels = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

        $this->testing = Labeled::quick($samples, $labels);

        $this->estimator = $this->createMock(RobustZScore::class);

        $this->estimator->method('type')->willReturn(RobustZScore::DETECTOR);

        $this->estimator->method('predict')->willReturn([
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        ]);

        $this->report = new OutlierRatio();
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(OutlierRatio::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $outcome = [
            'ratio' => 0.1111111111111111,
            'proportion' => 0.1,
            'percentage' => 10.0,
            'outliers' => 1,
            'inliers' => 9,
            'cardinality' => 10,
        ];

        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($outcome, $result);
    }
}
