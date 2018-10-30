<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use PHPUnit\Framework\TestCase;

class VMeasureTest extends TestCase
{
    protected $metric;

    protected $estimator;

    protected $testing;

    public function setUp()
    {
        $samples = [[], [], [], [], []];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->testing = Labeled::quick($samples, $labels);

        $this->estimator = $this->createMock(KMeans::class);

        $this->estimator->method('type')->willReturn(KMeans::CLUSTERER);

        $this->estimator->method('predict')->willReturn([
            1, 2, 2, 1, 2,
        ]);

        $this->metric = new VMeasure();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(VMeasure::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_get_range()
    {
        $this->assertEquals([0, 1], $this->metric->range());
    }

    public function test_score_predictions()
    {
        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertEquals(0.5833333333513888, $score);
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
