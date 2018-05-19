<?php

use Rubix\Engine\Metrics\Validation\RSquared;
use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Regression;
use Rubix\Engine\Estimators\Predictions\Prediction;
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
        $this->assertInstanceOf(Regression::class, $this->metric);
        $this->assertInstanceOf(Validation::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $predictions = [
            new Prediction(9), new Prediction(15),
            new Prediction(9), new Prediction(12),
            new Prediction(8),
        ];

        $outcomes = [10, 10, 6, 14, 8];

        $score = $this->metric->score($predictions, $outcomes);

        $this->assertEquals(2.2, $score, '', 5);
    }
}
