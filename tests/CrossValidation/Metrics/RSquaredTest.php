<?php

namespace Rubix\Tests\CrossValidation\Metrics;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\CrossValidation\Metrics\Metric;
use PHPUnit\Framework\TestCase;

class RSquaredTest extends TestCase
{
    protected $metric;

    protected $estimator;

    protected $testing;

    protected $outcome;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []], [10, 10, 6, 14, 8]);

        $this->estimator = $this->createMock(Ridge::class);

        $this->estimator->method('type')->willReturn(Ridge::REGRESSOR);

        $this->estimator->method('predict')->willReturn([
            9, 15, 9, 12, 8,
        ]);

        $this->metric = new RSquared();

        $this->outcome = -0.10795454542387639;
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(RSquared::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_get_range()
    {
        $this->assertEquals([-INF, 1], $this->metric->range());
    }

    public function test_score_predictions()
    {
        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertEquals($this->outcome, $score, '', 1e-8);
    }

    public function test_within_range()
    {
        list($min, $max) = $this->metric->range();

        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertThat($score, $this->logicalAnd(
            $this->greaterThanOrEqual($min), $this->lessThanOrEqual($max))
        );
    }
}
