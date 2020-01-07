<?php

namespace Rubix\ML\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\WildGuess;
use Rubix\ML\Other\Strategies\Continuous;
use PHPUnit\Framework\TestCase;

class WildGuessTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Strategies\WildGuess
     */
    protected $strategy;

    public function setUp() : void
    {
        $this->strategy = new WildGuess();
    }

    public function test_build_strategy() : void
    {
        $this->assertInstanceOf(WildGuess::class, $this->strategy);
        $this->assertInstanceOf(Continuous::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess() : void
    {
        $this->strategy->fit([1, 2, 3, 4, 5]);

        $guess = $this->strategy->guess();

        $this->assertThat(
            $guess,
            $this->logicalAnd(
                $this->greaterThanOrEqual(1),
                $this->lessThanOrEqual(5)
            )
        );
    }
}
