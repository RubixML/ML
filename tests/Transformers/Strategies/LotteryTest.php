<?php

use Rubix\Engine\Strategies\Strategy;
use Rubix\Engine\Strategies\Lottery;
use PHPUnit\Framework\TestCase;

class LotteryTest extends TestCase
{
    protected $values;

    protected $strategy;

    public function setUp()
    {
        $this->values = ['a', 'a', 'b', 'a', 'c'];

        $this->strategy = new Lottery();
    }

    public function test_build_local_celebrity_strategy()
    {
        $this->assertInstanceOf(Lottery::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $this->strategy->fit($this->values);

        $value = $this->strategy->guess();

        $this->assertContains($value, $this->values);
    }
}
