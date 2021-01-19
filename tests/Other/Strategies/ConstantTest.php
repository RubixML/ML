<?php

namespace Rubix\ML\Tests\Other\Strategies;

use Rubix\ML\DataType;
use Rubix\ML\Other\Strategies\Constant;
use Rubix\ML\Other\Strategies\Strategy;
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
        $this->strategy = new Constant(42);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Constant::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(DataType::continuous(), $this->strategy->type());
    }

    /**
     * @test
     */
    public function fitGuess() : void
    {
        $this->strategy->fit([]);

        $this->assertTrue($this->strategy->fitted());

        $guess = $this->strategy->guess();

        $this->assertEquals(42, $guess);
    }

    protected function assertPreConditions() : void
    {
        $this->assertTrue($this->strategy->fitted());
    }
}
