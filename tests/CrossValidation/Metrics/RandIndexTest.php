<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\RandIndex;
use PHPUnit\Framework\TestCase;

class RandIndexTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new RandIndex();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(RandIndex::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);

        $this->assertNotEmpty(array_filter($this->metric->range(), 'is_numeric'));
        $this->assertNotEmpty(array_filter($this->metric->compatibility(), 'is_int'));
    }

    public function test_score_predictions()
    {
        $predictions = [1, 2, 2, 1, 2,];

        $labels = ['lamb', 'lamb', 'wolf', 'lamb', 'wolf'];

        [$min, $max] = $this->metric->range();

        $score = $this->metric->score($predictions, $labels);

        $this->assertEquals(0.16666666666666663, $score);

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );
    }
}
