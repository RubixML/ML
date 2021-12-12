<?php

namespace Rubix\ML\Tests\CrossValidation\Reports;

use Rubix\ML\Report;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Reports\ReportGenerator;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Reports
 * @covers \Rubix\ML\CrossValidation\Reports\ConfusionMatrix
 */
class ConfusionMatrixTest extends TestCase
{
    /**
     * @var \Rubix\ML\CrossValidation\Reports\ConfusionMatrix
     */
    protected $report;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->report = new ConfusionMatrix();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(ConfusionMatrix::class, $this->report);
        $this->assertInstanceOf(ReportGenerator::class, $this->report);
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
            EstimatorType::classifier(),
            EstimatorType::anomalyDetector(),
        ];

        $this->assertEquals($expected, $this->report->compatibility());
    }

    /**
     * @test
     * @dataProvider generateProvider
     *
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @param mixed[] $expected
     */
    public function generate(array $predictions, array $labels, array $expected) : void
    {
        $result = $this->report->generate($predictions, $labels);

        $this->assertInstanceOf(Report::class, $result);
        $this->assertEquals($expected, $result->toArray());
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function generateProvider() : Generator
    {
        yield [
            ['wolf', 'lamb', 'wolf', 'lamb', 'wolf', 'lamb', 'lamb'],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf', 'lamb', 'wolf'],
            [
                'wolf' => [
                    'wolf' => 2,
                    'lamb' => 1,
                ],
                'lamb' => [
                    'wolf' => 2,
                    'lamb' => 2,
                ],
            ],
        ];
    }
}
