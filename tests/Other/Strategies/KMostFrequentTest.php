<?php

namespace Rubix\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Categorical;
use Rubix\ML\Other\Strategies\KMostFrequent;
use PHPUnit\Framework\TestCase;

class KMostFrequentTest extends TestCase
{
    protected $values;

    protected $strategy;

    public function setUp()
    {
        $this->values = ['a', 'a', 'b', 'b', 'c'];

        $this->strategy = new KMostFrequent(2);
    }

    public function test_build_strategy()
    {
        $this->assertInstanceOf(KMostFrequent::class, $this->strategy);
        $this->assertInstanceOf(Categorical::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_make_guess()
    {
        $this->strategy->fit($this->values);

        $value = $this->strategy->guess();

        $this->assertContains($value, $this->values);
    }
}
