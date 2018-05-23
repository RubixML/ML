<?php

use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Regression;
use Rubix\Engine\Metrics\Validation\MeanSquaredError;
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
        $this->assertInstanceOf(Validation::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $predictions = [9, 15, 9, 12, 8];

        $outcomes = [10, 10, 6, 14, 8];

        $score = $this->metric->score($predictions, $outcomes);

        $this->assertEquals(-8, $score, '', 5);
    }
}
