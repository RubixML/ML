<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tuple;
use Rubix\ML\CrossValidation\Metrics\ROCAUC;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Metrics')]
#[CoversClass(ROCAUC::class)]
class ROCAUCTest extends TestCase
{
    protected ROCAUC $metric;

    /**
     * @return Generator<array>
     */
    public static function scoreProvider() : Generator
    {
        yield [
            [
                ['A' => 0.9, 'B' => 0.1],
                ['A' => 0.8, 'B' => 0.2],
                ['A' => 0.2, 'B' => 0.8],
                ['A' => 0.1, 'B' => 0.9],
            ],
            ['A', 'A', 'B', 'B'],
            1.0,
        ];

        yield [
            [
                ['A' => 0.1, 'B' => 0.9],
                ['A' => 0.2, 'B' => 0.8],
                ['A' => 0.8, 'B' => 0.2],
                ['A' => 0.9, 'B' => 0.1],
            ],
            ['A', 'A', 'B', 'B'],
            0.0,
        ];

        yield [
            [
                ['A' => 0.5, 'B' => 0.5],
                ['A' => 0.5, 'B' => 0.5],
                ['A' => 0.5, 'B' => 0.5],
                ['A' => 0.5, 'B' => 0.5],
            ],
            ['A', 'A', 'B', 'B'],
            0.5,
        ];

        yield [
            [
                ['A' => 0.4, 'B' => 0.6],
                ['A' => 0.6, 'B' => 0.4],
                ['A' => 0.5, 'B' => 0.5],
                ['A' => 0.4, 'B' => 0.6],
            ],
            ['A', 'A', 'B', 'B'],
            0.625,
        ];

        yield [
            [
                ['A' => 0.7, 'B' => 0.2, 'C' => 0.1],
                ['A' => 0.6, 'B' => 0.3, 'C' => 0.1],
                ['A' => 0.3, 'B' => 0.5, 'C' => 0.2],
                ['A' => 0.1, 'B' => 0.6, 'C' => 0.3],
            ],
            ['A', 'A', 'B', 'C'],
            1.0,
        ];
    }

    protected function setUp() : void
    {
        $this->metric = new ROCAUC('A');
    }

    #[Test]
    public function range() : void
    {
        $tuple = $this->metric->range();

        $this->assertInstanceOf(Tuple::class, $tuple);
        $this->assertCount(2, $tuple);
        $this->assertGreaterThan($tuple[0], $tuple[1]);
    }

    /**
     * @param list<array<string,int|float>> $probabilities
     * @param list<string|int> $labels
     * @param float $expected
     */
    #[DataProvider('scoreProvider')]
    #[Test]
    public function score(array $probabilities, array $labels, float $expected) : void
    {
        [$min, $max] = $this->metric->range()->list();

        $score = $this->metric->score(
            probabilities: $probabilities,
            labels: $labels
        );

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );

        $this->assertEqualsWithDelta($expected, $score, 1e-8);
    }

    #[Test]
    public function scoreOnEmptySet() : void
    {
        $score = $this->metric->score(probabilities: [], labels: []);

        $this->assertEqualsWithDelta(0.0, $score, 1e-8);
    }

    #[Test]
    public function scoreWithMissingPositiveClass() : void
    {
        $probabilities = [
            ['A' => 0.3, 'B' => 0.7],
            ['A' => 0.4, 'B' => 0.6],
        ];

        $score = $this->metric->score(probabilities: $probabilities, labels: ['B', 'B']);

        $this->assertEqualsWithDelta(0.0, $score, 1e-8);
    }

    #[Test]
    public function scoreWithIntegerClasses() : void
    {
        $metric = new ROCAUC(0);

        $probabilities = [
            [0 => 0.9, 1 => 0.1],
            [0 => 0.8, 1 => 0.2],
            [0 => 0.2, 1 => 0.8],
            [0 => 0.1, 1 => 0.9],
        ];

        $score = $metric->score(probabilities: $probabilities, labels: [0, 0, 1, 1]);

        $this->assertEqualsWithDelta(1.0, $score, 1e-8);
    }
}
