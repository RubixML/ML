<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Initializers\LeCun;
use PHPUnit\Framework\TestCase;

#[Group('Initializers')]
#[CoversClass(LeCun::class)]
class LeCunTest extends TestCase
{
    protected LeCun $initializer;

    protected function setUp() : void
    {
        $this->initializer = new LeCun();
    }

    public function testInitialize() : void
    {
        $w = $this->initializer->initialize(fanIn: 4, fanOut: 3);

        $this->assertSame([3, 4], $w->shape());
    }
}
