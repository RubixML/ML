<?php

use MathPHP\LinearAlgebra\Matrix;
use Rubix\Engine\NeuralNet\ActivationFunctions\SoftPlus;
use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class SoftPlusTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5]]);

        $this->activationFunction = new SoftPlus();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(SoftPlus::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $activations = $this->activationFunction->compute($this->input);

        $this->assertEquals(1.3132616875182228, $activations[0][0]);
        $this->assertEquals(0.4740769841801067, $activations[1][0]);
    }

    public function test_differentiate()
    {
        $activations = $this->activationFunction->compute($this->input);

        $slopes = $this->activationFunction->differentiate($this->input, $activations);

        $this->assertEquals(0.7880584423829144, $slopes[0][0]);
        $this->assertEquals(0.6163482688094494, $slopes[1][0]);
    }
}
