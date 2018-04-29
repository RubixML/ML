<?php

use Rubix\Engine\Metrics\Error;
use Rubix\Engine\Metrics\Metric;
use Rubix\Engine\Metrics\RSquared;
use PHPUnit\Framework\TestCase;

class RSquaredTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new RSquared();
    }

    public function test_build_r_squared_metric()
    {
        $this->assertInstanceOf(RSquared::class, $this->metric);
        $this->assertInstanceOf(Error::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $predictions = [9, 11, 5, 12, 8];
        $outcomes = [10, 10, 6, 14, 8];

        $this->assertEquals(0.8011363636363636, $this->metric->score($predictions, $outcomes));
    }
}
