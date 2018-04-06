<?php

use Rubix\Engine\Preprocessors\Strategies\FuzzyMedian;
use PHPUnit\Framework\TestCase;

class FuzzyMedianTest extends TestCase
{
    protected $strategy;

    public function setUp()
    {
        $this->strategy = new FuzzyMedian();
    }

    public function test_build_strategy()
    {
        $this->assertInstanceOf(FuzzyMedian::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $data = [1, 2, 3, 4];

        $value = $this->strategy->guess($data);

        $this->assertTrue($value < 3 && $value > 2);
    }
}
