<?php

namespace Rubix\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Lottery;
use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Categorical;
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
        $this->assertInstanceOf(Categorical::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_make_guess()
    {
        $this->strategy->fit($this->values);

        $value = $this->strategy->guess();

        $this->assertContains($value, $this->values);
    }
}
