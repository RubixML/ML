<?php

use Rubix\Engine\Datasets\Labeled;
use Rubix\Tests\Helpers\MockRegressor;
use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Regression;
use Rubix\Engine\Metrics\Validation\MeanSquaredError;
use PHPUnit\Framework\TestCase;

class MeanSquaredErrorTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []], [10, 10, 6, 14, 8]);

        $this->estimator = new MockRegressor([9, 15, 9, 12, 8]);

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
        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertEquals(-8, $score, '', 5);
    }
}
