<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Tests\F1Score;
use Rubix\Engine\Tests\Loggers\BlackHole;
use PHPUnit\Framework\TestCase;

class F1ScoreTest extends TestCase
{
    protected $test;

    public function setUp()
    {
        $this->test = new F1Score(new BlackHole());
    }

    public function test_build_f1_score_test()
    {
        $this->assertInstanceOf(F1Score::class, $this->test);
    }

    public function test_score_predictions()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];
        $outcomes = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->assertEquals(0.5833333259722223, $this->test->score($predictions, $outcomes));
    }
}
