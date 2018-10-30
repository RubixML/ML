<?php

namespace Rubix\ML\Tests\Reports;

use Rubix\ML\Reports\Report;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\Classifiers\KNearestNeighbors;
use PHPUnit\Framework\TestCase;

class PredictionSpeedTest extends TestCase
{
    protected $report;

    protected $testing;

    protected $estimator;

    public function setUp()
    {
        $samples = [[], [], [], [], []];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->testing = Labeled::quick($samples, $labels);

        $this->estimator = $this->createMock(KNearestNeighbors::class);

        $this->estimator->method('predict')->willReturn([
            'wolf', 'lamb', 'wolf', 'lamb', 'wolf',
        ]);

        $this->report = new PredictionSpeed();
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(PredictionSpeed::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }
}
