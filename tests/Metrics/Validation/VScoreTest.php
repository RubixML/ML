<?php

use Rubix\Engine\Datasets\Labeled;
use Rubix\Tests\Helpers\MockClusterer;
use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Clustering;
use Rubix\Engine\Metrics\Validation\VMeasure;
use PHPUnit\Framework\TestCase;

class VMeasureTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

        $this->estimator = new MockClusterer([1, 2, 2, 1, 2]);

        $this->metric = new VMeasure();
    }

    public function test_build_v_score_metric()
    {
        $this->assertInstanceOf(VMeasure::class, $this->metric);
        $this->assertInstanceOf(Clustering::class, $this->metric);
        $this->assertInstanceOf(Validation::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertEquals(0.5833333280555556, $score);
    }
}
