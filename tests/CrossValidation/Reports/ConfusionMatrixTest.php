<?php

namespace Rubix\ML\Tests\CrossValidation\Reports;

use Rubix\ML\CrossValidation\Reports\Report;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use PHPUnit\Framework\TestCase;
use Generator;

class ConfusionMatrixTest extends TestCase
{
    protected $report;

    public function setUp()
    {
        $this->report = new ConfusionMatrix();
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(ConfusionMatrix::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    /**
     * @dataProvider generate_report_provider
     */
    public function test_generate_report(array $predictions, array $labels, array $expected)
    {
        $result = $this->report->generate($predictions, $labels);

        $this->assertEquals($expected, $result);
    }

    public function generate_report_provider() : Generator
    {
        yield [
            ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            [
                'wolf' => [
                    'wolf' => 2,
                    'lamb' => 1,
                ],
                'lamb' => [
                    'wolf' => 1,
                    'lamb' => 1,
                ],
            ],
        ];
    }
}
