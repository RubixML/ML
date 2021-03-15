<?php

namespace Rubix\ML\Tests\Strategies;

use Rubix\ML\DataType;
use Rubix\ML\Strategies\Prior;
use Rubix\ML\Strategies\Strategy;
use PHPUnit\Framework\TestCase;

/**
 * @group Strategies
 * @covers \Rubix\ML\Strategies\Prior
 */
class PriorTest extends TestCase
{
    /**
     * @var \Rubix\ML\Strategies\Prior
     */
    protected $strategy;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->strategy = new Prior();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Prior::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(DataType::categorical(), $this->strategy->type());
    }

    /**
     * @test
     */
    public function fitGuess() : void
    {
        $values = ['a', 'a', 'b', 'a', 'c'];

        $this->strategy->fit($values);

        $this->assertTrue($this->strategy->fitted());

        $value = $this->strategy->guess();

        $this->assertContains($value, $values);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->strategy->fitted());
    }
}
