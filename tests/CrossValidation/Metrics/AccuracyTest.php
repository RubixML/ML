<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;
use Generator;

class AccuracyTest extends TestCase
{
    /**
     * @var \Rubix\ML\CrossValidation\Metrics\Accuracy
     */
    protected $metric;

    public function setUp() : void
    {
        $this->metric = new Accuracy();
    }

    public function test_build_metric() : void
    {
        $this->assertInstanceOf(Accuracy::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);

        $this->assertNotEmpty(array_filter($this->metric->range(), 'is_numeric'));
        $this->assertNotEmpty(array_filter($this->metric->compatibility(), 'is_int'));
    }

    /**
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @param float $expected
     *
     * @dataProvider score_class_provider
     */
    public function test_score_class(array $predictions, array $labels, float $expected) : void
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
    public function score_class_provider() : Generator
    {
        yield [
            ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            0.6,
        ];

        yield [
            ['wolf', 'wolf', 'lamb', 'lamb', 'lamb'],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            0.0,
        ];

        yield [
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            1.0,
        ];
    }

    /**
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @param float $expected
     *
     * @dataProvider score_anomaly_provider
     */
    public function test_score_anomaly(array $predictions, array $labels, float $expected) : void
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
    public function score_anomaly_provider() : Generator
    {
        yield [
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            0.8,
        ];

        yield [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            1.0,
        ];

        yield [
            [1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0],
            0.0,
        ];
    }
}
