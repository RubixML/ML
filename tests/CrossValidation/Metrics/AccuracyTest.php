<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;

class AccuracyTest extends TestCase
{
    protected $metric;

    protected $estimator;

    protected $testing;

    public function setUp()
    {
        $samples = [[], [], [], [], []];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->testing = Labeled::quick($samples, $labels);

        $this->estimator = $this->createMock(KNearestNeighbors::class);

        $this->estimator->method('type')->willReturn(KNearestNeighbors::CLASSIFIER);

        $this->estimator->method('predict')->willReturn([
            'wolf', 'lamb', 'wolf', 'lamb', 'wolf',
        ]);

        $this->metric = new Accuracy();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(Accuracy::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_get_range()
    {
        $this->assertEquals([0, 1], $this->metric->range());
    }

    public function test_score_predictions()
    {
        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertEquals(0.6, $score);
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
