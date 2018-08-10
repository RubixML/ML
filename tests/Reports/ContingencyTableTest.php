<?php

namespace Rubix\Tests\Reports;

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
        $this->testing = new Labeled([[], [], [], [], []],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

        $this->estimator = $this->createMock(KMeans::class);

        $this->estimator->method('type')->willReturn(KMeans::CLUSTERER);

        $this->estimator->method('predict')->willReturn([
            1, 2, 2, 1, 2,
        ]);

        $this->report = new ContingencyTable();

        $this->outcome = [
            1 => [
                'wolf' => 1,
                'lamb' => 1,
            ],
            2 => [
                'wolf' => 2,
                'lamb' => 1,
            ],
        ];
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(ContingencyTable::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($this->outcome, $result);
    }
}
