<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Tests\Informedness;
use Rubix\Engine\Tests\Loggers\BlackHole;
use PHPUnit\Framework\TestCase;

class InformednessTest extends TestCase
{
    protected $test;

    public function setUp()
    {
        $this->test = new Informedness(new BlackHole());
    }

    public function test_build_informedness_test()
    {
        $this->assertInstanceOf(Informedness::class, $this->test);
    }

    public function test_score_predictions()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];
        $outcomes = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->assertEquals(0.16666666194444446, $this->test->score($predictions, $outcomes));
    }
}
