<?php

namespace Rubix\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Strategy;
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

    public function test_build_blurry_mean_strategy()
    {
        $this->assertInstanceOf(BlurryMedian::class, $this->strategy);
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
