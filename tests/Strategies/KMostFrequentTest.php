<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Strategies;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
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

    #[Test]
    public function preConditions() : void
    {
        $this->assertFalse($this->strategy->fitted());
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(DataType::categorical(), $this->strategy->type());
    }

    #[Test]
    public function fitGuess() : void
    {
        $values = ['a', 'a', 'b', 'b', 'c'];

        $this->strategy->fit($values);

        $this->assertTrue($this->strategy->fitted());

        $value = $this->strategy->guess();

        $this->assertContains($value, $values);
    }
}
