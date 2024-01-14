<?php

namespace Rubix\ML\Tests\Strategies;

use Rubix\ML\Strategies\Strategy;
use Rubix\ML\Strategies\Percentile;
use PHPUnit\Framework\TestCase;

/**
 * @group Strategies
 * @covers \Rubix\ML\Strategies\Percentile
 */
class PercentileTest extends TestCase
{
    /**
     * @var Percentile
     */
    protected $strategy;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->strategy = new Percentile(50.0);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Percentile::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    /**
     * @test
     */
    public function fitGuess() : void
    {
        $this->strategy->fit([1, 2, 3, 4, 5]);

        $guess = $this->strategy->guess();

        $this->assertEquals(3, $guess);
    }
}
