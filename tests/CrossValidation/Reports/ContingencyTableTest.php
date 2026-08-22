<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\CrossValidation\Reports;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\EstimatorType;
use Rubix\ML\Report;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Reports')]
#[CoversClass(ContingencyTable::class)]
class ContingencyTableTest extends TestCase
{
    protected ContingencyTable $report;

    /**
     * @return Generator<array>
     */
    public static function generateProvider() : Generator
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

    protected function setUp() : void
    {
        $this->report = new ContingencyTable();
    }

    public function testCompatibility() : void
    {
        $expected = [
            EstimatorType::clusterer(),
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
