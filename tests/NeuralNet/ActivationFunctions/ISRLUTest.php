<?php

namespace Rubix\Tests\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ISRLU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class ISRLUTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5]]);

        $this->activationFunction = new ISRLU(1.0);
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(ISRLU::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $activations = $this->activationFunction->compute($this->input);

        $this->assertEquals(1.0, $activations[0][0]);
        $this->assertEquals(-0.4472135954999579, $activations[1][0]);
    }

    public function test_differentiate()
    {
        $activations = $this->activationFunction->compute($this->input);

        $slopes = $this->activationFunction->differentiate($this->input, $activations);

        $this->assertEquals(1.0, $slopes[0][0]);
        $this->assertEquals(-0.08944271909999157, $slopes[1][0]);
    }
}
