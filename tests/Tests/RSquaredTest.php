<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Tests\RSquared;
use Rubix\Engine\Tests\Loggers\BlackHole;
use PHPUnit\Framework\TestCase;

class RSquaredTest extends TestCase
{
    protected $test;

    public function setUp()
    {
        $this->test = new RSquared(new BlackHole());
    }

    public function test_build_r_squared_test()
    {
        $this->assertInstanceOf(RSquared::class, $this->test);
    }

    public function test_score_predictions()
    {
        $predictions = [9, 15, 9, 12, 8];
        $outcomes = [10, 10, 6, 14, 8];

        $this->assertEquals(0.02985074626865658, $this->test->score($predictions, $outcomes));
    }
}
