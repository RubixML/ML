<?php

use Rubix\Engine\Metrics\Metric;
use Rubix\Engine\Metrics\F1Score;
use Rubix\Engine\Metrics\Classification;
use PHPUnit\Framework\TestCase;

class F1ScoreTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new F1Score();
    }

    public function test_build_f1_score_metric()
    {
        $this->assertInstanceOf(F1Score::class, $this->metric);
        $this->assertInstanceOf(Classification::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'wolf', 'wolf', 'wolf', 'lamb', 'wolf', 'wolf', 'wolf'];
        $outcomes = ['wolf', 'lamb', 'wolf', 'wolf', 'wolf', 'lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->assertEquals(0.8666666766666667, $this->metric->score($predictions, $outcomes));
    }
}
