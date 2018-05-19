<?php

use Rubix\Engine\Metrics\Validation\F1Score;
use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Classification;
use Rubix\Engine\Estimators\Predictions\Prediction;
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
        $this->assertInstanceOf(Validation::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $predictions = [
            new Prediction('wolf'), new Prediction('lamb'),
            new Prediction('wolf'), new Prediction('lamb'),
            new Prediction('wolf'),
        ];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $score = $this->metric->score($predictions, $labels);

        $this->assertEquals(0.5833333404166667, $score);
    }
}
