<?php

namespace Rubix\ML\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Categorical;
use Rubix\ML\Other\Strategies\KMostFrequent;
use PHPUnit\Framework\TestCase;

class KMostFrequentTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Strategies\KMostFrequent
     */
    protected $strategy;

    public function setUp() : void
    {
        $this->strategy = new KMostFrequent(2);
    }

    public function test_build_strategy() : void
    {
        $this->assertInstanceOf(KMostFrequent::class, $this->strategy);
        $this->assertInstanceOf(Categorical::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_classes_guess() : void
    {
        $values = ['a', 'a', 'b', 'b', 'c'];

        $this->strategy->fit($values);

        $value = $this->strategy->guess();

        $this->assertContains($value, $values);
    }
}
