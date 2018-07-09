<?php

namespace Rubix\Tests\Reports;

use Rubix\ML\Reports\Report;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\Tests\Helpers\MockClassifier;
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

    public function test_build_report()
    {
        $this->assertInstanceOf(PredictionSpeed::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }
}
