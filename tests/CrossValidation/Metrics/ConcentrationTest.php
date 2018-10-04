<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\Concentration;
use PHPUnit\Framework\TestCase;

class ConcentrationTest extends TestCase
{
    protected $metric;

    protected $estimator;

    protected $testing;

    protected $outcome;

    public function setUp()
    {
        $this->testing = new Unlabeled([
            [10, 40], [100, 150], [90, 150], [20, 30], [110, 140],
        ]);

        $this->estimator = $this->createMock(KMeans::class);

        $this->estimator->method('type')->willReturn(KMeans::CLUSTERER);

        $this->estimator->method('predict')->willReturn([
            1, 2, 2, 1, 2,
        ]);

        $this->metric = new Concentration();

        $this->outcome = 193.36363636363635;
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(Concentration::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_get_range()
    {
        $this->assertEquals([-INF, INF], $this->metric->range());
    }

    public function test_score_predictions()
    {
        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertEquals($this->outcome, $score);
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
