<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Strategies;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Strategies\Prior;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Strategies')]
#[CoversClass(Prior::class)]
class PriorTest extends TestCase
{
    protected Prior $strategy;

    protected function setUp() : void
    {
        $this->strategy = new Prior();
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(DataType::categorical(), $this->strategy->type());
    }

    #[Test]
    public function fitGuess() : void
    {
        $values = ['a', 'a', 'b', 'a', 'c'];

        $this->strategy->fit($values);

        $this->assertTrue($this->strategy->fitted());

        $value = $this->strategy->guess();

        $this->assertContains($value, $values);
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertFalse($this->strategy->fitted());
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
