<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\NeuralNet\Parameter;
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
        $parameter = $this->initializer->initialize([3, 4]);

        $expected = [
            [4.8, 4.8, 4.8, 4.8],
            [4.8, 4.8, 4.8, 4.8],
            [4.8, 4.8, 4.8, 4.8],
        ];

        $this->assertInstanceOf(Parameter::class, $parameter);
        $this->assertInstanceOf(Matrix::class, $parameter->param());
        $this->assertEquals([3, 4], $parameter->param()->shape());
        $this->assertEquals($expected, $parameter->param()->asArray());
    }

    #[Test]
    public function initializeVector() : void
    {
        $parameter = $this->initializer->initialize([4]);

        $expected = [4.8, 4.8, 4.8, 4.8];

        $this->assertInstanceOf(Parameter::class, $parameter);
        $this->assertInstanceOf(Vector::class, $parameter->param());
        $this->assertEquals([4], $parameter->param()->shape());
        $this->assertEquals($expected, $parameter->param()->asArray());
    }
}
