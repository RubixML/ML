<?php

namespace Rubix\Tests\Reports;

use Rubix\ML\Reports\Report;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Classifiers\KNearestNeighbors;
use PHPUnit\Framework\TestCase;

class ConfusionMatrixTest extends TestCase
{
    protected $report;

    protected $testing;

    protected $estimator;

    protected $outcome;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

        $this->estimator = $this->createMock(KNearestNeighbors::class);

        $this->estimator->method('type')->willReturn(KNearestNeighbors::CLASSIFIER);

        $this->estimator->method('predict')->willReturn([
            'wolf', 'lamb', 'wolf', 'lamb', 'wolf',
        ]);

        $this->report = new ConfusionMatrix(['wolf', 'lamb']);

        $this->outcome = [
            'wolf' => [
                'wolf' => 2,
                'lamb' => 1,
            ],
            'lamb' => [
                'wolf' => 1,
                'lamb' => 1,
            ],
        ];
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(ConfusionMatrix::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($this->outcome, $result);
    }
}
