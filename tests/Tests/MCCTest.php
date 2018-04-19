<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Tests\MCC;
use Rubix\Engine\Tests\Loggers\BlackHole;
use PHPUnit\Framework\TestCase;

class MCCTest extends TestCase
{
    protected $test;

    public function setUp()
    {
        $this->test = new MCC(new BlackHole());
    }

    public function test_build_mcc_test()
    {
        $this->assertInstanceOf(MCC::class, $this->test);
    }

    public function test_score_predictions()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];
        $outcomes = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->assertEquals(0.16666667666666665, $this->test->score($predictions, $outcomes));
    }
}
