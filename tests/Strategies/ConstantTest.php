<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Strategies;

use PHPUnit\Framework\Attributes\CoversClass;
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

    public function testAssertPreConditions() : void
    {
        $this->assertTrue($this->strategy->fitted());
    }

    public function testType() : void
    {
        $this->assertEquals(DataType::continuous(), $this->strategy->type());
    }

    public function testFitGuess() : void
    {
        $this->strategy->fit([]);

        $this->assertTrue($this->strategy->fitted());

        $guess = $this->strategy->guess();

        $this->assertEquals(42, $guess);
    }
}
