<?php

use Rubix\Engine\Metrics\MeanAbsoluteError;
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
    }

    public function test_score_predictions()
    {
        $predictions = [9, 15, 9, 12, 8];
        $outcomes = [10, 10, 6, 14, 8];

        $this->assertEquals(2.2, $this->metric->score($predictions, $outcomes));
    }
}
