<?php

namespace Rubix\ML\Tests\Strategies;

use Rubix\ML\DataType;
use Rubix\ML\Strategies\Strategy;
use Rubix\ML\Strategies\WildGuess;
use PHPUnit\Framework\TestCase;

/**
 * @group Strategies
 * @covers \Rubix\ML\Strategies\WildGuess
 */
class WildGuessTest extends TestCase
{
    /**
     * @var WildGuess
     */
    protected $strategy;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->strategy = new WildGuess();
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
        $this->assertInstanceOf(WildGuess::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(DataType::continuous(), $this->strategy->type());
    }

    /**
     * @test
     */
    public function fitGuess() : void
    {
        $this->strategy->fit([1, 2, 3, 4, 5]);

        $this->assertTrue($this->strategy->fitted());

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
