<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Initializers\Xavier1;
use PHPUnit\Framework\TestCase;

#[Group('Initializers')]
#[CoversClass(Xavier1::class)]
class Xavier1Test extends TestCase
{
    protected Xavier1 $initializer;

    protected function setUp() : void
    {
        $this->initializer = new Xavier1();
    }

    public function testInitialize() : void
    {
        $w = $this->initializer->initialize(fanIn: 4, fanOut: 3);

        $this->assertSame([3, 4], $w->shape());
    }
}
