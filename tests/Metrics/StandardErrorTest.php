<?php

use Rubix\Engine\Metrics\StandardError;
use PHPUnit\Framework\TestCase;

class StandardErrorTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new StandardError();
    }

    public function test_build_standard_error_metric()
    {
        $this->assertInstanceOf(StandardError::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $predictions = [9, 15, 9, 12, 8];
        $outcomes = [10, 10, 6, 14, 8];

        $this->assertEquals(0.7694153624668537, $this->metric->score($predictions, $outcomes));
    }
}
