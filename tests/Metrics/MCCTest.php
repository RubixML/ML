<?php

use Rubix\Engine\Metrics\MCC;
use Rubix\Engine\Metrics\Metric;
use Rubix\Engine\Metrics\Classification;
use PHPUnit\Framework\TestCase;

class MCCTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new MCC();
    }

    public function test_build_mcc_test()
    {
        $this->assertInstanceOf(MCC::class, $this->metric);
        $this->assertInstanceOf(Classification::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];
        $outcomes = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->assertEquals(0.16666666648148146, $this->metric->score($predictions, $outcomes));
    }
}
