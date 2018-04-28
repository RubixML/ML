<?php

use Rubix\Engine\Transformers\Strategies\Strategy;
use Rubix\Engine\Transformers\Strategies\RandomMedian;
use PHPUnit\Framework\TestCase;

class RandomMedianTest extends TestCase
{
    protected $strategy;

    public function setUp()
    {
        $this->strategy = new RandomMedian();
    }

    public function test_build_random_median_strategy()
    {
        $this->assertInstanceOf(RandomMedian::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $data = [1, 2, 3, 4];

        $value = $this->strategy->guess($data);

        $this->assertContains($value, $data);
    }
}
