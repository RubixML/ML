<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Tuple;
use Rubix\ML\CrossValidation\Metrics\ProbabilisticMetric;
use Rubix\ML\CrossValidation\Metrics\ProbabilisticDistance;

/**
 * @group Metrics
 * @covers \Rubix\ML\CrossValidation\Metrics\ProbabilisticDistance
 */
class ProbabilisticDistanceTest extends TestCase
{
    /**
     * @var \Rubix\ML\CrossValidation\Metrics\ProbabilisticDistance
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->metric = new ProbabilisticDistance();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(ProbabilisticMetric::class, $this->metric);
        $this->assertInstanceOf(ProbabilisticDistance::class, $this->metric);
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
    public function score() : void
    {
        [$min, $max] = $this->metric->range()->list();

        $score = $this->metric->score(
            [
                ['U' => 0.7, 'O' => 0.3],
                ['U' => 0.4, 'O' => 0.6],
                ['U' => 0.2, 'O' => 0.8],
            ],
            ['U', 'O', 'U']
        );

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );

        $this->assertEquals(-0.5, $score);
    }

    /**
     * @test
     */
    public function scoreEmptyProbabilities() : void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->metric->score([], ['U', 'O', 'U']);
    }

    /**
     * @test
     */
    public function scoreLabelsAreIncompatibleWithProbabilities() : void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->metric->score(
            [
                ['U' => 0.7, 'O' => 0.3],
            ],
            ['U', 'O', 'U']
        );
    }
}
