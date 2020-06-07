<?php

namespace Rubix\ML\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Constant;
use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Continuous;
use PHPUnit\Framework\TestCase;

/**
 * @group Strategies
 * @covers \Rubix\ML\Other\Strategies\Constant
 */
class ConstantTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Strategies\Constant
     */
    protected $strategy;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->strategy = new Constant(17.0);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Constant::class, $this->strategy);
        $this->assertInstanceOf(Continuous::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    /**
     * @test
     */
    public function fitGuess() : void
    {
        $this->strategy->fit([]);

        $guess = $this->strategy->guess();

        $this->assertEquals(17.0, $guess);
    }
}
