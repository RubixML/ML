<?php

namespace Rubix\ML\Tests\NeuralNet;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use PHPUnit\Framework\TestCase;

class ParameterTest extends TestCase
{
    protected $parameter;

    public function setUp()
    {
        $this->parameter = new Parameter(Matrix::quick([[5]]));
    }

    public function test_build_parameter()
    {
        $this->assertInstanceOf(Parameter::class, $this->parameter);
    }

    public function test_get_w()
    {
        $this->assertInstanceOf(Matrix::class, $this->parameter->w());
        $this->assertEquals(5, $this->parameter->w()[0][0]);
    }

    public function test_update()
    {
        $this->assertEquals(5, $this->parameter->w()[0][0]);

        $this->parameter->update(Matrix::quick([[-1]]));

        $this->assertEquals(6, $this->parameter->w()[0][0]);
    }
}
