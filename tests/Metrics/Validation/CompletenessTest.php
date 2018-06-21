<?php

use Rubix\ML\Datasets\Labeled;
use Rubix\Tests\Helpers\MockClusterer;
use Rubix\ML\Metrics\Validation\Validation;
use Rubix\ML\Metrics\Validation\Clustering;
use Rubix\ML\Metrics\Validation\Completeness;
use PHPUnit\Framework\TestCase;

class CompletenessTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

        $this->estimator = new MockClusterer([1, 2, 2, 1, 2]);

        $this->metric = new Completeness();
    }

    public function test_build_completeness_metric()
    {
        $this->assertInstanceOf(Completeness::class, $this->metric);
        $this->assertInstanceOf(Clustering::class, $this->metric);
        $this->assertInstanceOf(Validation::class, $this->metric);
    }
    
    public function test_get_range()
    {
        $this->assertEquals([0, 1], $this->metric->range());
    }

    public function test_score_predictions()
    {
        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertEquals(0.5833333280555556, $score);
    }
}
