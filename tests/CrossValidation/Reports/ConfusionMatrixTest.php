<?php

namespace Rubix\ML\Tests\CrossValidation\Reports;

use Rubix\ML\CrossValidation\Reports\Report;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use PHPUnit\Framework\TestCase;
use Generator;

class ConfusionMatrixTest extends TestCase
{
    /**
     * @var \Rubix\ML\CrossValidation\Reports\ConfusionMatrix
     */
    protected $report;

    public function setUp() : void
    {
        $this->report = new ConfusionMatrix();
    }

    public function test_build_report() : void
    {
        $this->assertInstanceOf(ConfusionMatrix::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    /**
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @param mixed[] $expected
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
