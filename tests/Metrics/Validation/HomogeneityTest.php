<?php

use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Clustering;
use Rubix\Engine\Metrics\Validation\Homogeneity;
use PHPUnit\Framework\TestCase;

class HomogeneityTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new Homogeneity();
    }

    public function test_build_homogeneity_metric()
    {
        $this->assertInstanceOf(Homogeneity::class, $this->metric);
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
