<?php

namespace Rubix\ML\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Constant;
use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Continuous;
use PHPUnit\Framework\TestCase;

class ConstantTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Strategies\Constant
     */
    protected $strategy;

    public function setUp() : void
    {
        $this->strategy = new Constant(17.);
    }

    public function test_build_strategy() : void
    {
        $this->assertInstanceOf(Constant::class, $this->strategy);
        $this->assertInstanceOf(Continuous::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess() : void
    {
        $this->strategy->fit([]);

        $guess = $this->strategy->guess();

        $this->assertEquals(17., $guess);
    }
}
