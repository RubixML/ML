<?php

use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Regression;
use Rubix\Engine\Metrics\Validation\MeanAbsoluteError;
use PHPUnit\Framework\TestCase;

class MeanAbsoluteErrorTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new MeanAbsoluteError();
    }

    public function test_build_mean_absolute_error_metric()
    {
        $this->assertInstanceOf(MeanAbsoluteError::class, $this->metric);
        $this->assertInstanceOf(Regression::class, $this->metric);
        $this->assertInstanceOf(Validation::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $predictions = [9, 15, 9, 12, 8];

        $outcomes = [10, 10, 6, 14, 8];

        $score = $this->metric->score($predictions, $outcomes);

        $this->assertEquals(2.2, $score, '', 5);
    }
}
