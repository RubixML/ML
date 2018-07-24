<?php

namespace Rubix\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Continuous;
use Rubix\ML\Other\Strategies\BlurryMedian;
use PHPUnit\Framework\TestCase;

class BlurryMedianTest extends TestCase
{
    protected $values;

    protected $strategy;

    public function setUp()
    {
        $this->values = [1, 2, 3, 4, 5];

        $this->strategy = new BlurryMedian();
    }

    public function test_build_strategy()
    {
        $this->assertInstanceOf(BlurryMedian::class, $this->strategy);
        $this->assertInstanceOf(Continuous::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_make_guess()
    {
        $this->strategy->fit($this->values);

        $guess = $this->strategy->guess();

        $this->assertThat($guess, $this->logicalAnd(
            $this->greaterThanOrEqual(min($this->values)),
            $this->lessThanOrEqual(max($this->values)))
        );
    }
}
