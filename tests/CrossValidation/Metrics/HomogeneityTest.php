<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\Homogeneity;
use PHPUnit\Framework\TestCase;
use Generator;

class HomogeneityTest extends TestCase
{
    /**
     * @var \Rubix\ML\CrossValidation\Metrics\Homogeneity
     */
    protected $metric;

    public function setUp() : void
    {
        $this->metric = new Homogeneity();
    }

    public function test_build_metric() : void
    {
        $this->assertInstanceOf(Homogeneity::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);

        $this->assertNotEmpty(array_filter($this->metric->range(), 'is_numeric'));
        $this->assertNotEmpty(array_filter($this->metric->compatibility(), 'is_int'));
    }

    /**
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
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
            [0, 1, 1, 0, 1],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            0.5833333333333333,
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
            1.0,
        ];

        yield [
            [0, 0, 0, 0, 0],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            0.6,
        ];
    }
}
