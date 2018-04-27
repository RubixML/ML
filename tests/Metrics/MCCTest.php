<?php

use Rubix\Engine\Metrics\MCC;
use Rubix\Engine\Datasets\Dataset;
use PHPUnit\Framework\TestCase;

class MCCTest extends TestCase
{
    protected $test;

    public function setUp()
    {
        $this->test = new MCC();
    }

    public function test_build_mcc_test()
    {
        $this->assertInstanceOf(MCC::class, $this->test);
    }

    public function test_score_predictions()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];
        $outcomes = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->assertEquals(0.16666666648148146, $this->test->score($predictions, $outcomes));
    }
}
