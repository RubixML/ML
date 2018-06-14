<?php

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class SigmoidTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5]]);

        $this->activationFunction = new Sigmoid();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(Sigmoid::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $activations = $this->activationFunction->compute($this->input);

        $this->assertEquals(0.7310585786300049, $activations[0][0]);
        $this->assertEquals(0.3775406687981454, $activations[1][0]);
    }

    public function test_differentiate()
    {
        $activations = $this->activationFunction->compute($this->input);

        $slopes = $this->activationFunction->differentiate($this->input, $activations);

        $this->assertEquals(0.19661193324148185, $slopes[0][0]);
        $this->assertEquals(0.2350037122015945, $slopes[1][0]);
    }
}
