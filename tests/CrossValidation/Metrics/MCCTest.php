<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\CrossValidation\Metrics\MCC;
use Rubix\ML\CrossValidation\Metrics\Metric;
use PHPUnit\Framework\TestCase;

class MCCTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new MCC();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(MCC::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_get_range()
    {
        $this->assertEquals([-1, 1], $this->metric->range());
    }

    public function test_score_predictions()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        list($min, $max) = $this->metric->range();

        $score = $this->metric->score($predictions, $labels);

        $this->assertEquals(0.16666666668055555, $score);

        $this->assertThat($score, $this->logicalAnd(
            $this->greaterThanOrEqual($min), $this->lessThanOrEqual($max))
        );
    }
}
