<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\CrossValidation\Metrics\Metric;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Metrics
 * @covers \Rubix\ML\CrossValidation\Metrics\RMSE
 */
class RMSETest extends TestCase
{
    /**
     * @var RMSE
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->metric = new RMSE();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(RMSE::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
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
        [$min, $max] = $this->metric->range()->list();

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
     * @return \Generator<mixed[]>
     */
    public function scoreProvider() : Generator
    {
        yield [
            [7, 9.5, -20, -500, .079],
            [10, 10.0, 6, -1400, .08],
            -402.6624516890046,
        ];

        yield [
            [0, 0, 0, 0, 0],
            [10, 10.0, 6, -1400, .08],
            -626.1367273048276,
        ];

        yield [
            [10, 10.0, 6, -1400, .08],
            [10, 10.0, 6, -1400, .08],
            0.0,
        ];
    }
}
