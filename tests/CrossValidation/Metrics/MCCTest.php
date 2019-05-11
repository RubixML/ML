<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\CrossValidation\Metrics\MCC;
use Rubix\ML\CrossValidation\Metrics\Metric;
use PHPUnit\Framework\TestCase;
use Generator;

class MCCTest extends TestCase
{
    protected const LABELS = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

    protected $metric;

    public function setUp()
    {
        $this->metric = new MCC();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(MCC::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);

        $this->assertNotEmpty(array_filter($this->metric->range(), 'is_numeric'));
        $this->assertNotEmpty(array_filter($this->metric->compatibility(), 'is_int'));
    }

    /**
     * @dataProvider score_provider
     */
    public function test_score(array $predictions, float $expected)
    {
        [$min, $max] = $this->metric->range();

        $score = $this->metric->score($predictions, self::LABELS);

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );

        $this->assertEquals($expected, $score);
    }

    public function score_provider() : Generator
    {
        yield [['wolf', 'lamb', 'wolf', 'lamb', 'wolf'], 0.16666666666666666];
        yield [['wolf', 'wolf', 'lamb', 'lamb', 'lamb'], -1.0];
        yield [['lamb', 'lamb', 'wolf', 'wolf', 'wolf'], 1.0];
    }
}
