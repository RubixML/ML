<?php

namespace Rubix\ML\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Continuous;
use Rubix\ML\Other\Strategies\Percentile;
use PHPUnit\Framework\TestCase;

class PercentileTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Strategies\Percentile
     */
    protected $strategy;

    public function setUp() : void
    {
        $this->strategy = new Percentile(50.);
    }

    public function test_build_strategy() : void
    {
        $this->assertInstanceOf(Percentile::class, $this->strategy);
        $this->assertInstanceOf(Continuous::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess() : void
    {
        $this->strategy->fit([1, 2, 3, 4, 5]);

        $guess = $this->strategy->guess();

        $this->assertEquals(3, $guess);
    }
}
