<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Strategies;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Strategies\KMostFrequent;
use PHPUnit\Framework\TestCase;

#[Group('Strategies')]
#[CoversClass(KMostFrequent::class)]
class KMostFrequentTest extends TestCase
{
    protected KMostFrequent $strategy;

    protected function setUp() : void
    {
        $this->strategy = new KMostFrequent(2);
    }

    public function testAssertPreConditions() : void
    {
        $this->assertFalse($this->strategy->fitted());
    }

    public function testType() : void
    {
        $this->assertEquals(DataType::categorical(), $this->strategy->type());
    }

    public function testFitGuess() : void
    {
        $values = ['a', 'a', 'b', 'b', 'c'];

        $this->strategy->fit($values);

        $this->assertTrue($this->strategy->fitted());

        $value = $this->strategy->guess();

        $this->assertContains($value, $values);
    }
}
