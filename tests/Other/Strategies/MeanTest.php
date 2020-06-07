<?php

namespace Rubix\ML\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Mean;
use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Continuous;
use PHPUnit\Framework\TestCase;

/**
 * @group Strategies
 * @covers \Rubix\ML\Other\Strategies\Mean
 */
class MeanTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Strategies\Mean
     */
    protected $strategy;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->strategy = new Mean();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Mean::class, $this->strategy);
        $this->assertInstanceOf(Continuous::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    /**
     * @test
     */
    public function fitGuess() : void
    {
        $this->strategy->fit([1, 2, 3, 4, 5]);

        $guess = $this->strategy->guess();

        $this->assertEquals(3.0, $guess);
    }
}
