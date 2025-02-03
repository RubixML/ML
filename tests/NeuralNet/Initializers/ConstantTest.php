<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Initializers\Constant;
use PHPUnit\Framework\TestCase;

#[Group('Initializers')]
#[CoversClass(Constant::class)]
class ConstantTest extends TestCase
{
    protected Constant $initializer;

    protected function setUp() : void
    {
        $this->initializer = new Constant(4.8);
    }

    public function testInitialize() : void
    {
        $w = $this->initializer->initialize(fanIn: 4, fanOut: 3);

        $expected = [
            [4.8, 4.8, 4.8, 4.8],
            [4.8, 4.8, 4.8, 4.8],
            [4.8, 4.8, 4.8, 4.8],
        ];

        $this->assertSame([3, 4], $w->shape());
        $this->assertSame($expected, $w->asArray());
    }
}
