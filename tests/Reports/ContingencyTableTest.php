<?php

namespace Rubix\ML\Tests\Reports;

use Rubix\ML\Reports\Report;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Reports\ContingencyTable;
use PHPUnit\Framework\TestCase;

class ContingencyTableTest extends TestCase
{
    protected $report;

    protected $testing;

    protected $estimator;

    protected $outcome;

    public function setUp()
    {
        $samples = [[], [], [], [], []];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->testing = Labeled::quick($samples, $labels);

        $this->estimator = $this->createMock(KMeans::class);

        $this->estimator->method('type')->willReturn(KMeans::CLUSTERER);

        $this->estimator->method('predict')->willReturn([
            1, 2, 2, 1, 2,
        ]);

        $this->report = new ContingencyTable();
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(ContingencyTable::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $outcome = [
            1 => [
                'wolf' => 1,
                'lamb' => 1,
            ],
            2 => [
                'wolf' => 2,
                'lamb' => 1,
            ],
        ];

        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($outcome, $result);
    }
}
