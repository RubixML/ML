<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Tests\StandardError;
use Rubix\Engine\Tests\Loggers\BlackHole;
use PHPUnit\Framework\TestCase;

class StandardErrorTest extends TestCase
{
    protected $test;

    public function setUp()
    {
        $this->test = new StandardError(new BlackHole());
    }

    public function test_build_standard_error_test()
    {
        $this->assertInstanceOf(StandardError::class, $this->test);
    }

    public function test_score_predictions()
    {
        $predictions = [9, 15, 9, 12, 8];
        $outcomes = [10, 10, 6, 14, 8];

        $this->assertEquals(0.7694153624668537, $this->test->score($predictions, $outcomes));
    }
}
