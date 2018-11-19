<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\Completeness;
use PHPUnit\Framework\TestCase;

class CompletenessTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new Completeness();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(Completeness::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_get_range()
    {
        $this->assertEquals([0, 1], $this->metric->range());
    }

    public function test_score_predictions()
    {
        $predictions = [1, 2, 2, 1, 2];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        list($min, $max) = $this->metric->range();

        $score = $this->metric->score($predictions, $labels);

        $this->assertEquals(0.5833333333513888, $score);

        $this->assertThat($score, $this->logicalAnd(
            $this->greaterThanOrEqual($min), $this->lessThanOrEqual($max))
        );
    }
}
