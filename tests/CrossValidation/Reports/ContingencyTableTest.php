<?php

namespace Rubix\ML\Tests\CrossValidation\Reports;

use Rubix\ML\CrossValidation\Reports\Report;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;
use PHPUnit\Framework\TestCase;

class ContingencyTableTest extends TestCase
{
    protected $report;

    public function setUp()
    {
        $this->report = new ContingencyTable();
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(ContingencyTable::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $predictions = [1, 2, 2, 1, 2];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

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

        $result = $this->report->generate($predictions, $labels);

        $this->assertEquals($outcome, $result);
    }
}
