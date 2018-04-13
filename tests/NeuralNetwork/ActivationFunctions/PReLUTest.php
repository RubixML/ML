<?php

use Rubix\Engine\NeuralNetwork\ActivationFunctions\PReLU;
use Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class PReLUTest extends TestCase
{
    protected $activationFunction;

    public function setUp()
    {
        $this->activationFunction = new PReLU(0.01);
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(PReLU::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $this->assertEquals(1.0, $this->activationFunction->compute(1.0));
        $this->assertEquals(-0.005, $this->activationFunction->compute(-0.5));
    }

    public function test_differentiate()
    {
        $this->assertEquals(1.0, $this->activationFunction->differentiate(1.0, 1.0));
        $this->assertEquals(0.01, $this->activationFunction->differentiate(-0.5, -0.005));
    }
}
