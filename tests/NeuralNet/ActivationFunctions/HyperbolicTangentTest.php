<?php

use MathPHP\LinearAlgebra\Matrix;
use Rubix\Engine\NeuralNet\ActivationFunctions\HyperbolicTangent;
use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class HyperbolicTangentTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5]]);

        $this->activationFunction = new HyperbolicTangent();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(HyperbolicTangent::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $activations = $this->activationFunction->compute($this->input);

        $this->assertEquals(0.7615941559557649, $activations[0][0]);
        $this->assertEquals(-0.46211715726000974, $activations[1][0]);
    }

    public function test_differentiate()
    {
        $activations = $this->activationFunction->compute($this->input);

        $slopes = $this->activationFunction->differentiate($this->input, $activations);

        $this->assertEquals(0.41997434161402614, $slopes[0][0]);
        $this->assertEquals(0.7864477329659274, $slopes[1][0]);
    }
}
