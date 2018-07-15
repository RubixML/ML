<?php

namespace Rubix\Tests\CrossValidation\Metrics;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\CrossValidation\Metrics\Validation;
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;
use PHPUnit\Framework\TestCase;

class MeanAbsoluteErrorTest extends TestCase
{
    protected $metric;

    protected $estimator;

    protected $testing;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []], [10, 10, 6, 14, 8]);

        $this->estimator = $this->createMock(Ridge::class);

        $this->estimator->method('predict')->willReturn([
            9, 15, 9, 12, 8,
        ]);

        $this->metric = new MeanAbsoluteError();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(MeanAbsoluteError::class, $this->metric);
        $this->assertInstanceOf(Validation::class, $this->metric);
    }

    public function test_get_range()
    {
        $this->assertEquals([-INF, 0], $this->metric->range());
    }

    public function test_score_predictions()
    {
        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertEquals(2.2, $score, '', 5);
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
