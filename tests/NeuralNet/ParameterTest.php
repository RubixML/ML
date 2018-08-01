<?php

namespace Rubix\Tests\NeuralNet;

use Rubix\ML\NeuralNet\Parameter;
use MathPHP\LinearAlgebra\Matrix;
use PHPUnit\Framework\TestCase;

class ParameterTest extends TestCase
{
    protected $parameter;

    public function setUp()
    {
        $this->parameter = new Parameter(new Matrix([]));
    }

    public function test_build_parameter()
    {
        $this->assertInstanceOf(Parameter::class, $this->parameter);
    }

    public function test_get_w()
    {
        $this->assertInstanceOf(Matrix::class, $this->parameter->w());
    }
}
