<?php

use Rubix\Engine\Metrics\Validation\RMSError;
use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Regression;
use PHPUnit\Framework\TestCase;

class RMSErrorTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new RMSError();
    }

    public function test_build_rms_error_metric()
    {
        $this->assertInstanceOf(RMSError::class, $this->metric);
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
