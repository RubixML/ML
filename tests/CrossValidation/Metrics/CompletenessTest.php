<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\Completeness;
use PHPUnit\Framework\TestCase;
use Generator;

class CompletenessTest extends TestCase
{
    protected const LABELS = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

    protected $metric;

    public function setUp()
    {
        $this->metric = new Completeness();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(Completeness::class, $this->metric);
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
        yield [[0, 1, 1, 0, 1], 0.5833333333333333];
        yield [[0, 0, 1, 1, 1], 1.0];
        yield [[1, 1, 0, 0, 0], 1.0];
        yield [[0, 1, 2, 3, 4], 0.41666666666666663];
        yield [[0, 0, 0, 0, 0], 1.0];
    }
}
