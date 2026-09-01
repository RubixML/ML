<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\TestCase;

/**
 * @group Initializers
 * @covers \Rubix\ML\NeuralNet\Initializers\Constant
 */
class ConstantTest extends TestCase
{
    /**
     * @var Constant
     */
    protected $initializer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->initializer = new Constant(4.8);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Constant::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    /**
     * @test
     */
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
