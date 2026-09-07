<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Strategies;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Strategies\WildGuess;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Strategies')]
#[CoversClass(WildGuess::class)]
class WildGuessTest extends TestCase
{
    protected WildGuess $strategy;

    protected function setUp() : void
    {
        $this->strategy = new WildGuess();
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertFalse($this->strategy->fitted());
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(DataType::continuous(), $this->strategy->type());
    }

    #[Test]
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

    #[Test]
    public function guessThrowsWhenUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $this->strategy->guess();
    }

    #[Test]
    public function fitRejectsEmptySet() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->strategy->fit([]);
    }
}
