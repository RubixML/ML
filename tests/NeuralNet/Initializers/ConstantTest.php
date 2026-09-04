<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Initializers')]
#[CoversClass(Constant::class)]
class ConstantTest extends TestCase
{
    /**
     * @var Constant
     */
    protected Constant $initializer;

    protected function setUp() : void
    {
        $this->initializer = new Constant(4.8);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Constant::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    #[Test]
    public function initialize() : void
    {
        $w = $this->initializer->initialize(4, 3);

        $expected = [
            [4.8, 4.8, 4.8, 4.8],
            [4.8, 4.8, 4.8, 4.8],
            [4.8, 4.8, 4.8, 4.8],
        ];

        $this->assertInstanceOf(Matrix::class, $w);
        $this->assertEquals([3, 4], $w->shape());
        $this->assertEquals($expected, $w->asArray());
    }
}
