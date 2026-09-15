<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Initializers\Uniform;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Initializers')]
#[CoversClass(Uniform::class)]
class UniformTest extends TestCase
{
    /**
     * @var Uniform
     */
    protected Uniform $initializer;

    protected function setUp() : void
    {
        $this->initializer = new Uniform(0.05);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Uniform::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    #[Test]
    public function initialize() : void
    {
        $parameter = $this->initializer->initialize([3, 4]);

        $this->assertInstanceOf(Parameter::class, $parameter);
        $this->assertInstanceOf(Matrix::class, $parameter->param());
        $this->assertEquals([3, 4], $parameter->param()->shape());
    }

    #[Test]
    public function initializeVector() : void
    {
        $parameter = $this->initializer->initialize([4]);

        $this->assertInstanceOf(Parameter::class, $parameter);
        $this->assertInstanceOf(Vector::class, $parameter->param());
        $this->assertEquals([4], $parameter->param()->shape());
    }
}
