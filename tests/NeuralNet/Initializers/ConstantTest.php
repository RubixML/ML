<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\TestCase;

class ConstantTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Initializers\Constant
     */
    protected $initializer;

    public function setUp() : void
    {
        $this->initializer = new Constant(4.8);
    }

    public function test_build_layer() : void
    {
        $this->assertInstanceOf(Constant::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    public function test_initialize() : void
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
