<?php

namespace Rubix\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\WildGuess;
use PHPUnit\Framework\TestCase;

class WildGuessTest extends TestCase
{
    protected $values;

    protected $strategy;

    public function setUp()
    {
        $this->values = [1, 2, 3, 4, 5];

        $this->strategy = new WildGuess(2);
    }

    public function test_build_random_copy_paste_strategy()
    {
        $this->assertInstanceOf(WildGuess::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $this->strategy->fit($this->values);

        list($min, $max) = $this->strategy->range();

        $value = $this->strategy->guess();

        $this->assertThat($value, $this->logicalAnd(
            $this->greaterThanOrEqual($min), $this->lessThanOrEqual($max))
        );
    }
}
