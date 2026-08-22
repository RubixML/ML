<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tuple;
use Rubix\ML\CrossValidation\Metrics\ProbabilisticAccuracy;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Metrics')]
#[CoversClass(ProbabilisticAccuracy::class)]
class ProbabilisticAccuracyTest extends TestCase
{
    protected ProbabilisticAccuracy $metric;

    /**
     * @return Generator<array>
     */
    public static function scoreProvider() : Generator
    {
        yield [
            [
                ['U' => 0.7, 'O' => 0.3],
                ['U' => 0.4, 'O' => 0.6],
                ['U' => 0.2, 'O' => 0.8],
            ],
            ['U', 'O', 'U'],
            0.5,
        ];

        yield [
            [
                ['cat' => 0.6, 'frog' => 0.1, 'hamster' => 0.3],
                ['cat' => 0.1, 'frog' => 0.7, 'hamster' => 0.2],
                ['cat' => 0.2, 'frog' => 0.0, 'hamster' => 0.9],
                ['cat' => 0.5, 'frog' => 0.1, 'hamster' => 0.4],
                ['cat' => 0.0, 'frog' => 0.8, 'hamster' => 0.2],
            ],
            ['cat', 'frog', 'hamster', 'cat', 'frog'],
            0.7,
        ];

        yield [
            [
                [1 => 0.0, 2 => 0.0, 3 => 1.0],
                [1 => 0.0, 2 => 1.0, 3 => 0.0],
                [1 => 1.0, 2 => 0.0, 3 => 0.0],
            ],
            [3, 2, 1],
            1.0,
        ];

        yield [
            [
                [1 => 0.2, 2 => 0.8, 3 => 0.0],
                [1 => 0.5, 2 => 0.0, 3 => 0.5],
                [1 => 0.0, 2 => 1.0, 3 => 0.0],
            ],
            [3, 2, 1],
            0.0,
        ];

        yield [
            [
                ['yes' => 1.0, 'no' => 0.0],
                ['yes' => 0.0, 'no' => 1.0],
            ],
            ['no', 'yes'],
            0.0,
        ];

        yield [
            [
                ['yes' => 0.5, 'no' => 0.5],
                ['yes' => 0.5, 'no' => 0.5],
            ],
            ['no', 'yes'],
            0.5,
        ];
    }

    protected function setUp() : void
    {
        $this->metric = new ProbabilisticAccuracy();
    }

    public function testRange() : void
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
    public function testScore(array $probabilities, array $labels, float $expected) : void
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
}
