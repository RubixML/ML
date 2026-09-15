<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Initializers\Normal;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Initializers')]
#[CoversClass(Normal::class)]
class NormalTest extends TestCase
{
    /**
     * @var Normal
     */
    protected Normal $initializer;

    protected function setUp() : void
    {
        $this->initializer = new Normal(0.05);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Normal::class, $this->initializer);
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
