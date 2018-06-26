<?php

use Rubix\ML\Datasets\Labeled;
use Rubix\Tests\Helpers\MockDetector;
use Rubix\ML\CrossValidation\Reports\Report;
use Rubix\ML\CrossValidation\Reports\OutlierRatio;
use PHPUnit\Framework\TestCase;

class OutlierRatioTest extends TestCase
{
    protected $report;

    protected $estimator;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], [], [], [], [], [], []],
            [11, 12, 14, 40, 55, 12, 16, 10, 2, 7]);

        $this->estimator = new MockDetector([
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        ]);

        $this->report = new OutlierRatio();
    }

    public function test_build_outlier_ratio_report()
    {
        $this->assertInstanceOf(OutlierRatio::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $actual = [
            'outliers' => 1,
            'inliers' => 9,
            'ratio' => 0.1111111109876543,
            'cardinality' => 10,
        ];

        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($actual, $result);
    }
}
