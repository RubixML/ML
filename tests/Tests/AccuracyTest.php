<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\Tests\Loggers\BlackHole;
use PHPUnit\Framework\TestCase;

class AccuracyTest extends TestCase
{
    protected $test;

    public function setUp()
    {
        $this->test = new Accuracy(new BlackHole());
    }

    public function test_build_accuracy_test()
    {
        $this->assertInstanceOf(Accuracy::class, $this->test);
    }

    public function test_score_predictions()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];
        $outcomes = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->assertEquals(0.5833333333333333, $this->test->score($predictions, $outcomes));
    }
}
