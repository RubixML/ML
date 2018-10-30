<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\CrossValidation\Metrics\RMSError;
use Rubix\ML\CrossValidation\Metrics\Metric;
use PHPUnit\Framework\TestCase;

class RMSErrorTest extends TestCase
{
    const TOLERANCE = 1e-10;

    protected $metric;

    protected $estimator;

    protected $testing;

    public function setUp()
    {
        $samples = [[], [], [], [], []];

        $labels = [10, 10, 6, 14, 8];

        $this->testing = Labeled::quick($samples, $labels);

        $this->estimator = $this->createMock(Ridge::class);

        $this->estimator->method('type')->willReturn(Ridge::REGRESSOR);

        $this->estimator->method('predict')->willReturn([
            9, 15, 9, 12, 8,
        ]);

        $this->metric = new RMSError();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(RMSError::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_get_range()
    {
        $this->assertEquals([-INF, 0], $this->metric->range());
    }

    public function test_score_predictions()
    {
        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertEquals(-2.792848008753788, $score, '', self::TOLERANCE);
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
