<?php

use Rubix\Engine\NeuralNetwork\ActivationFunctions\Heaviside;
use Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class HeavisideTest extends TestCase
{
    protected $activationFunction;

    public function setUp()
    {
        $this->activationFunction = new Heaviside();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(Heaviside::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $this->assertEquals(1.0, $this->activationFunction->compute(1.0));
        $this->assertEquals(0.0, $this->activationFunction->compute(-0.5));
    }

    public function test_differentiate()
    {
        $this->assertEquals(0.0, $this->activationFunction->differentiate(1.0, 1.0));
        $this->assertEquals(0.0, $this->activationFunction->differentiate(-0.5, 0.0));
    }
}
