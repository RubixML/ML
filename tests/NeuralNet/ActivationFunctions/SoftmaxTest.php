<?php

use MathPHP\LinearAlgebra\Matrix;
use Rubix\Engine\NeuralNet\ActivationFunctions\Softmax;
use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class SoftmaxTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5]]);

        $this->activationFunction = new Softmax();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(Softmax::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $activations = $this->activationFunction->compute($this->input);

        $this->assertEquals(0.8175744737346343, $activations[0][0]);
        $this->assertEquals(0.18242552325767722, $activations[1][0]);
    }

    public function test_differentiate()
    {
        $activations = $this->activationFunction->compute($this->input);

        $slopes = $this->activationFunction->differentiate($this->input, $activations);

        $this->assertEquals(0.14914645363217005, $slopes[0][0]);
        $this->assertEquals(0.14914645172183988, $slopes[1][0]);
    }
}
