<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Initializers\LeCun;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Initializers')]
#[CoversClass(LeCun::class)]
class LeCunTest extends TestCase
{
    /**
     * @var LeCun
     */
    protected LeCun $initializer;

    protected function setUp() : void
    {
        $this->initializer = new LeCun();
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(LeCun::class, $this->initializer);
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
