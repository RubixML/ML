<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Initializers\Uniform;
use PHPUnit\Framework\TestCase;

#[Group('Initializers')]
#[CoversClass(Uniform::class)]
class UniformTest extends TestCase
{
    protected Uniform $initializer;

    protected function setUp() : void
    {
        $this->initializer = new Uniform(0.05);
    }

    public function testInitialize() : void
    {
        $w = $this->initializer->initialize(fanIn: 4, fanOut: 3);

        $this->assertSame([3, 4], $w->shape());
    }
}
