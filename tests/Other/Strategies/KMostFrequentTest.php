<?php

namespace Rubix\ML\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Categorical;
use Rubix\ML\Other\Strategies\KMostFrequent;
use PHPUnit\Framework\TestCase;

/**
 * @group Strategies
 * @covers \Rubix\ML\Other\Strategies\KMostFrequent
 */
class KMostFrequentTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Strategies\KMostFrequent
     */
    protected $strategy;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->strategy = new KMostFrequent(2);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(KMostFrequent::class, $this->strategy);
        $this->assertInstanceOf(Categorical::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    /**
     * @test
     */
    public function fitGuess() : void
    {
        $values = ['a', 'a', 'b', 'b', 'c'];

        $this->strategy->fit($values);

        $value = $this->strategy->guess();

        $this->assertContains($value, $values);
    }
}
