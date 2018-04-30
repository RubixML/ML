<?php

use Rubix\Engine\Metrics\Metric;
use Rubix\Engine\Metrics\Accuracy;
use Rubix\Engine\Metrics\Classification;
use PHPUnit\Framework\TestCase;

class AccuracyTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new Accuracy();
    }

    public function test_build_accuracy_metric()
    {
        $this->assertInstanceOf(Accuracy::class, $this->metric);
        $this->assertInstanceOf(Classification::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];
        $outcomes = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->assertEquals(0.5999999988, $this->metric->score($predictions, $outcomes));
    }
}
