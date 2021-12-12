<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\RandIndex;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Metrics
 * @covers \Rubix\ML\CrossValidation\Metrics\RandIndex
 */
class RandIndexTest extends TestCase
{
    /**
     * @var \Rubix\ML\CrossValidation\Metrics\RandIndex
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->metric = new RandIndex();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(RandIndex::class, $this->metric);
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
            EstimatorType::clusterer(),
        ];

        $this->assertEquals($expected, $this->metric->compatibility());
    }

    /**
     * @test
     * @dataProvider scoreProvider
     *
     * @param (int|string)[] $predictions
     * @param (int|string)[] $labels
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
            [0, 1, 1, 0, 1],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            -0.25000000000000006,
        ];

        yield [
            [0, 0, 1, 1, 1],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            1.0,
        ];

        yield [
            [1, 1, 0, 0, 0],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            1.0,
        ];

        yield [
            [0, 1, 2, 3, 4],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            0.0,
        ];

        yield [
            [0, 0, 0, 0, 0],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            0.0,
        ];
    }
}
