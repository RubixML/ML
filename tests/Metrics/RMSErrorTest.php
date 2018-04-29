<?php

use Rubix\Engine\Metrics\Error;
use Rubix\Engine\Metrics\Metric;
use Rubix\Engine\Metrics\RMSError;
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
        $this->assertInstanceOf(Error::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $predictions = [9, 15, 9, 12, 8];
        $outcomes = [10, 10, 6, 14, 8];

        $this->assertEquals(2.7928480087537886, $this->metric->score($predictions, $outcomes));
    }
}
