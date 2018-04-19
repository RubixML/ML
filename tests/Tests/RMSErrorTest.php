<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Tests\RMSError;
use Rubix\Engine\Tests\Loggers\BlackHole;
use PHPUnit\Framework\TestCase;

class RMSErrorTest extends TestCase
{
    protected $test;

    public function setUp()
    {
        $this->test = new RMSError(new BlackHole());
    }

    public function test_build_rms_error_test()
    {
        $this->assertInstanceOf(RMSError::class, $this->test);
    }

    public function test_score_predictions()
    {
        $predictions = [9, 15, 9, 12, 8];
        $outcomes = [10, 10, 6, 14, 8];

        $this->assertEquals(2.7928480087537886, $this->test->score($predictions, $outcomes));
    }
}
