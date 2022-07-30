<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\CrossValidation\Metrics\ProbabilisticMetric;
use Rubix\ML\CrossValidation\Metrics\ProbabilisticAccuracy;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Metrics
 * @covers \Rubix\ML\CrossValidation\Metrics\ProbabilisticAccuracy
 */
class ProbabilisticAccuracyTest extends TestCase
{
    /**
     * @var \Rubix\ML\CrossValidation\Metrics\ProbabilisticAccuracy
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->metric = new ProbabilisticAccuracy();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(ProbabilisticMetric::class, $this->metric);
        $this->assertInstanceOf(ProbabilisticAccuracy::class, $this->metric);
    }

    /**
     * @test
     */
    public function range() : void
    {
        $tuple = $this->metric->range();

        $this->assertInstanceOf(Tuple::class, $tuple);
        $this->assertCount(2, $tuple);
        $this->assertGreaterThan($tuple[0], $tuple[1]);
    }

    /**
     * @test
     * @dataProvider scoreProvider
     *
     * @param list<array<string,int|float>> $probabilities
     * @param list<string|int> $labels
     * @param float $expected
     */
    public function score(array $probabilities, array $labels, float $expected) : void
    {
        [$min, $max] = $this->metric->range()->list();

        $score = $this->metric->score($probabilities, $labels);

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );

        $this->assertEqualsWithDelta($expected, $score, 1e-8);
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function scoreProvider() : Generator
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
}
