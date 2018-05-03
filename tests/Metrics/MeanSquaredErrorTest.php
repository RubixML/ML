<?php

use Rubix\Engine\Metrics\Metric;
use Rubix\Engine\Metrics\Regression;
use Rubix\Engine\Metrics\MeanSquaredError;
use PHPUnit\Framework\TestCase;

class MeanSquaredErrorTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new MeanSquaredError();
    }

    public function test_build_mean_squared_error_metric()
    {
        $this->assertInstanceOf(MeanSquaredError::class, $this->metric);
        $this->assertInstanceOf(Regression::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $predictions = [9, 15, 9, 12, 8];
        $outcomes = [10, 10, 6, 14, 8];

        $this->assertEquals(7.8, $this->metric->score($predictions, $outcomes));
    }
}
