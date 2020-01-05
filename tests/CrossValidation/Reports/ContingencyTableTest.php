<?php

namespace Rubix\ML\Tests\CrossValidation\Reports;

use Rubix\ML\CrossValidation\Reports\Report;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;
use PHPUnit\Framework\TestCase;
use Generator;

class ContingencyTableTest extends TestCase
{
    /**
     * @var \Rubix\ML\CrossValidation\Reports\ContingencyTable
     */
    protected $report;

    public function setUp() : void
    {
        $this->report = new ContingencyTable();
    }

    public function test_build_report() : void
    {
        $this->assertInstanceOf(ContingencyTable::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    /**
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @param array[] $expected
     *
     * @dataProvider generate_report_provider
     */
    public function test_generate_report(array $predictions, array $labels, array $expected) : void
    {
        $result = $this->report->generate($predictions, $labels);

        $this->assertEquals($expected, $result);
    }

    /**
     * @return \Generator<array>
     */
    public function generate_report_provider() : Generator
    {
        yield [
            [0, 1, 1, 0, 1],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            [
                0 => [
                    'wolf' => 1,
                    'lamb' => 1,
                ],
                1 => [
                    'wolf' => 2,
                    'lamb' => 1,
                ],
            ],
        ];
    }
}
