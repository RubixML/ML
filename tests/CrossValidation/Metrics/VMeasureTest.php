<?php

namespace Rubix\Tests\CrossValidation\Metrics;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\CrossValidation\Metrics\Validation;
use PHPUnit\Framework\TestCase;

class VMeasureTest extends TestCase
{
    protected $metric;

    protected $estimator;

    protected $testing;

    protected $outcome;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

        $this->estimator = $this->createMock(KMeans::class);

        $this->estimator->method('predict')->willReturn([
            1, 2, 2, 1, 2,
        ]);

        $this->metric = new VMeasure();

        $this->outcome = 0.5833333351388889;
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(VMeasure::class, $this->metric);
        $this->assertInstanceOf(Validation::class, $this->metric);
    }

    public function test_get_range()
    {
        $this->assertEquals([0, 1], $this->metric->range());
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
