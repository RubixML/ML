<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Initializers\Normal;
use PHPUnit\Framework\TestCase;

#[Group('Initializers')]
#[CoversClass(Normal::class)]
class NormalTest extends TestCase
{
    protected Normal $initializer;

    protected function setUp() : void
    {
        $this->initializer = new Normal(0.05);
    }

    /**
     * @test
     */
    public function testInitialize() : void
    {
        $w = $this->initializer->initialize(fanIn: 4, fanOut: 3);

        $this->assertSame([3, 4], $w->shape());
    }
}
