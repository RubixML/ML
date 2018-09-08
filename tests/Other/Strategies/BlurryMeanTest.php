<?php

namespace Rubix\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Continuous;
use Rubix\ML\Other\Strategies\BlurryMean;
use PHPUnit\Framework\TestCase;

class BlurryMeanTest extends TestCase
{
    protected $values;

    protected $range;

    protected $strategy;

    public function setUp()
    {
        $this->values = [1, 2, 3, 4, 5];

        $this->range = [0, 6];

        $this->strategy = new BlurryMean(0.3);
    }

    public function test_build_strategy()
    {
        $this->assertInstanceOf(BlurryMean::class, $this->strategy);
        $this->assertInstanceOf(Continuous::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_make_guess()
    {
        $this->strategy->fit($this->values);

        $guess = $this->strategy->guess();

        $this->assertThat($guess, $this->logicalAnd(
            $this->greaterThanOrEqual($this->range[0]),
            $this->lessThanOrEqual($this->range[1]))
        );
    }
}
