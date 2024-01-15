<?php

namespace Rubix\ML\Tests\Strategies;

use Rubix\ML\DataType;
use Rubix\ML\Strategies\Strategy;
use Rubix\ML\Strategies\KMostFrequent;
use PHPUnit\Framework\TestCase;

/**
 * @group Strategies
 * @covers \Rubix\ML\Strategies\KMostFrequent
 */
class KMostFrequentTest extends TestCase
{
    /**
     * @var KMostFrequent
     */
    protected $strategy;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->strategy = new KMostFrequent(2);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->strategy->fitted());
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(KMostFrequent::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(DataType::categorical(), $this->strategy->type());
    }

    /**
     * @test
     */
    public function fitGuess() : void
    {
        $values = ['a', 'a', 'b', 'b', 'c'];

        $this->strategy->fit($values);

        $this->assertTrue($this->strategy->fitted());

        $value = $this->strategy->guess();

        $this->assertContains($value, $values);
    }
}
