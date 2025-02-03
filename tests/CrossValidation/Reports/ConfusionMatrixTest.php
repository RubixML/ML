<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\CrossValidation\Reports;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Report;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Reports')]
#[CoversClass(ConfusionMatrix::class)]
class ConfusionMatrixTest extends TestCase
{
    protected ConfusionMatrix $report;

    /**
     * @return Generator<array>
     */
    public static function generateProvider() : Generator
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

    protected function setUp() : void
    {
        $this->report = new ConfusionMatrix();
    }

    public function testCompatibility() : void
    {
        $expected = [
            EstimatorType::classifier(),
            EstimatorType::anomalyDetector(),
        ];

        $this->assertEquals($expected, $this->report->compatibility());
    }

    /**
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @param array $expected
     */
    #[DataProvider('generateProvider')]
    public function testGenerate(array $predictions, array $labels, array $expected) : void
    {
        $result = $this->report->generate(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertInstanceOf(Report::class, $result);
        $this->assertEquals($expected, $result->toArray());
    }
}
