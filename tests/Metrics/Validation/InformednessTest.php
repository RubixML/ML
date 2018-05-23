<?php

use Rubix\Engine\Metrics\Validation\Informedness;
use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Classification;
use PHPUnit\Framework\TestCase;

class InformednessTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new Informedness();
    }

    public function test_build_mcc_test()
    {
        $this->assertInstanceOf(Informedness::class, $this->metric);
        $this->assertInstanceOf(Classification::class, $this->metric);
        $this->assertInstanceOf(Validation::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $score = $this->metric->score($predictions, $labels);

        $this->assertEquals(0.16666666111111106, $score);
    }
}
