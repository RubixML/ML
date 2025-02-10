<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Initializers\Xavier2;
use PHPUnit\Framework\TestCase;

#[Group('Initializers')]
#[CoversClass(Xavier2::class)]
class Xavier2Test extends TestCase
{
    protected Xavier2 $initializer;

    protected function setUp() : void
    {
        $this->initializer = new Xavier2();
    }

    public function testInitialize() : void
    {
        $w = $this->initializer->initialize(fanIn: 4, fanOut:  3);

        $this->assertSame([3, 4], $w->shape());
    }
}
