<?php

namespace Rubix\ML\Tests\Strategies;

use PHPUnit\Framework\TestCase;
use Rubix\ML\DataType;
use Rubix\ML\Strategies\Median;
use Rubix\ML\Strategies\Strategy;

/**
 * @group Strategies
 * @covers \Rubix\ML\Strategies\Median
 */
class MedianTest extends TestCase
{
    /**
     * @var Median
     */
    protected $strategy;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->strategy = new Median();
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->strategy->fitted());
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Median::class, $this->strategy);
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
        $this->strategy->fit([12, 3, 5]);

        $this->assertTrue($this->strategy->fitted());

        $guess = $this->strategy->guess();

        $this->assertEquals(5.0, $guess);
    }
}
