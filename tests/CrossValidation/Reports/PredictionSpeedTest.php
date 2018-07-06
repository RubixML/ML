<?php

namespace Rubix\Tests\CrossValidation\Reports;

use Rubix\ML\Datasets\Labeled;
use Rubix\Tests\Helpers\MockClassifier;
use Rubix\ML\Reports\Report;
use Rubix\ML\Reports\PredictionSpeed;
use PHPUnit\Framework\TestCase;

class PredictionSpeedTest extends TestCase
{
    protected $report;

    protected $testing;

    protected $estimator;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

        $this->estimator = new MockClassifier([
            'wolf', 'lamb', 'wolf', 'lamb', 'wolf'
        ]);

        $this->report = new PredictionSpeed();
    }

    public function test_build_prediction_speed_report()
    {
        $this->assertInstanceOf(PredictionSpeed::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }
}
