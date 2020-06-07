<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\MedianAbsoluteError;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Metrics
 * @covers \Rubix\ML\CrossValidation\Metrics\MedianAbsoluteError
 */
class MedianAbsoluteErrorTest extends TestCase
{
    /**
     * @var \Rubix\ML\CrossValidation\Metrics\MedianAbsoluteError
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->metric = new MedianAbsoluteError();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(MedianAbsoluteError::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    /**
     * @test
     */
    public function range() : void
    {
        $expected = [-INF, 0.0];

        $this->assertEquals($expected, $this->metric->range());
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
            EstimatorType::regressor(),
        ];

        $this->assertEquals($expected, $this->metric->compatibility());
    }

    /**
     * @test
     * @dataProvider scoreProvider
     *
     * @param (int|float)[] $predictions
     * @param (int|float)[] $labels
     * @param float $expected
     */
    public function score(array $predictions, array $labels, float $expected) : void
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
    public function scoreProvider() : Generator
    {
        yield [
            [7, 9.5, -20, -500, .079],
            [10, 10.0, 6, -1400, .08],
            -3.0,
        ];

        yield [
            [0, 0, 0, 0, 0],
            [10, 10.0, 6, -1400, .08],
            -10.0,
        ];

        yield [
            [10, 10.0, 6, -1400, .08],
            [10, 10.0, 6, -1400, .08],
            0.0,
        ];
    }
}
