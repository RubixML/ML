<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;
use PHPUnit\Framework\TestCase;
use Generator;

class MeanAbsoluteErrorTest extends TestCase
{
    /**
     * @var \Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError
     */
    protected $metric;

    public function setUp() : void
    {
        $this->metric = new MeanAbsoluteError();
    }

    public function test_build_metric() : void
    {
        $this->assertInstanceOf(MeanAbsoluteError::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);

        $this->assertNotEmpty(array_filter($this->metric->range(), 'is_numeric'));
        $this->assertNotEmpty(array_filter($this->metric->compatibility(), 'is_int'));
    }

    /**
     * @param (int|float)[] $predictions
     * @param (int|float)[] $labels
     * @param float $expected
     *
     * @dataProvider score_provider
     */
    public function test_score(array $predictions, array $labels, float $expected) : void
    {
        [$min, $max] = $this->metric->range();

        $score = $this->metric->score($predictions, $labels);

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );

        $this->assertEquals($expected, $score);
    }

    /**
     * @return \Generator<array>
     */
    public function score_provider() : Generator
    {
        yield [
            [7, 9.5, -20, -500, .079],
            [10, 10.0, 6, -1400, .08],
            -185.90019999999998,
        ];

        yield [
            [0, 0, 0, 0, 0],
            [10, 10.0, 6, -1400, .08],
            -285.216,
        ];

        yield [
            [10, 10.0, 6, -1400, .08],
            [10, 10.0, 6, -1400, .08],
            0.0,
        ];
    }
}
