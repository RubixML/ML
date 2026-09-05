<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Strategies;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Strategies\Constant;
use PHPUnit\Framework\TestCase;

#[Group('Strategies')]
#[CoversClass(Constant::class)]
class ConstantTest extends TestCase
{
    protected Constant $strategy;

    protected function setUp() : void
    {
        $this->strategy = new Constant(42);
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertTrue($this->strategy->fitted());
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(DataType::continuous(), $this->strategy->type());
    }

    #[Test]
    public function fitGuess() : void
    {
        $this->strategy->fit([]);

        $this->assertTrue($this->strategy->fitted());

        $guess = $this->strategy->guess();

        $this->assertEquals(42, $guess);
    }
}
