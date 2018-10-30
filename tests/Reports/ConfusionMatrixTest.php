<?php

namespace Rubix\ML\Tests\Reports;

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

    public function setUp()
    {
        $samples = [[], [], [], [], []];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->testing = Labeled::quick($samples, $labels);

        $this->estimator = $this->createMock(KNearestNeighbors::class);

        $this->estimator->method('type')->willReturn(KNearestNeighbors::CLASSIFIER);

        $this->estimator->method('predict')->willReturn([
            'wolf', 'lamb', 'wolf', 'lamb', 'wolf',
        ]);

        $this->report = new ConfusionMatrix(['wolf', 'lamb']);
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(ConfusionMatrix::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $outcome = [
            'wolf' => [
                'wolf' => 2,
                'lamb' => 1,
            ],
            'lamb' => [
                'wolf' => 1,
                'lamb' => 1,
            ],
        ];

        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($outcome, $result);
    }
}
