<?php

use Rubix\Engine\Datasets\Labeled;
use Rubix\Tests\Helpers\MockClassifier;
use Rubix\Engine\Metrics\Validation\Informedness;
use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Classification;
use PHPUnit\Framework\TestCase;

class InformednessTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

        $this->estimator = new MockClassifier([
            'wolf', 'lamb', 'wolf', 'lamb', 'wolf'
        ]);

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
        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertEquals(0.16666666111111106, $score);
    }
}
