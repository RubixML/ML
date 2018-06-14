<?php

use Rubix\ML\Datasets\Labeled;
use Rubix\Tests\Helpers\MockClassifier;
use Rubix\ML\Metrics\Validation\Accuracy;
use Rubix\ML\Metrics\Validation\Validation;
use Rubix\ML\Metrics\Validation\Classification;
use PHPUnit\Framework\TestCase;

class AccuracyTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

        $this->estimator = new MockClassifier([
            'wolf', 'lamb', 'wolf', 'lamb', 'wolf'
        ]);

        $this->metric = new Accuracy();
    }

    public function test_build_accuracy_metric()
    {
        $this->assertInstanceOf(Accuracy::class, $this->metric);
        $this->assertInstanceOf(Classification::class, $this->metric);
        $this->assertInstanceOf(Validation::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertEquals(0.5999999988, $score);
    }
}
