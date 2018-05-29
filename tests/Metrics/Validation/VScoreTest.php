<?php

use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Clustering;
use Rubix\Engine\Metrics\Validation\VMeasure;
use PHPUnit\Framework\TestCase;

class VMeasureTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
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
        $predictions = [1, 2, 2, 1, 2];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $score = $this->metric->score($predictions, $labels);

        $this->assertEquals(0.5833333280555556, $score);
    }
}
