<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Tests\MeanAbsoluteError;
use Rubix\Engine\Tests\Loggers\BlackHole;
use PHPUnit\Framework\TestCase;

class MeanAbsoluteErrorTest extends TestCase
{
    protected $test;

    public function setUp()
    {
        $this->test = new MeanAbsoluteError(new BlackHole());
    }

    public function test_build_mean_absolute_error_test()
    {
        $this->assertInstanceOf(MeanAbsoluteError::class, $this->test);
    }

    public function test_score_predictions()
    {
        $predictions = [9, 15, 9, 12, 8];
        $outcomes = [10, 10, 6, 14, 8];

        $this->assertEquals(2.2, $this->test->score($predictions, $outcomes));
    }
}
