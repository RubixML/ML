<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Strategies;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Strategies\Prior;
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

    public function testType() : void
    {
        $this->assertEquals(DataType::categorical(), $this->strategy->type());
    }

    public function testFitGuess() : void
    {
        $values = ['a', 'a', 'b', 'a', 'c'];

        $this->strategy->fit($values);

        $this->assertTrue($this->strategy->fitted());

        $value = $this->strategy->guess();

        $this->assertContains($value, $values);
    }

    protected function testAssertPreConditions() : void
    {
        $this->assertFalse($this->strategy->fitted());
    }
}
