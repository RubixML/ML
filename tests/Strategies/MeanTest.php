<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Strategies;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Strategies\Mean;
use PHPUnit\Framework\TestCase;

#[Group('Strategies')]
#[CoversClass(Mean::class)]
class MeanTest extends TestCase
{
    protected Mean $strategy;

    protected function setUp() : void
    {
        $this->strategy = new Mean();
    }

    public function testAssertPreConditions() : void
    {
        $this->assertFalse($this->strategy->fitted());
    }

    public function testType() : void
    {
        $this->assertEquals(DataType::continuous(), $this->strategy->type());
    }

    public function testFitGuess() : void
    {
        $this->strategy->fit([1, 2, 3, 4, 5]);

        $this->assertTrue($this->strategy->fitted());

        $guess = $this->strategy->guess();

        $this->assertEquals(3.0, $guess);
    }
}
