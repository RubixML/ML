<?php

use Rubix\Engine\NeuralNetwork\ActivationFunctions\Softsign;
use Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class SoftsignTest extends TestCase
{
    protected $activationFunction;

    public function setUp()
    {
        $this->activationFunction = new Softsign();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(Softsign::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $this->assertEquals(0.5, $this->activationFunction->compute(1.0));
        $this->assertEquals(-0.3333333333333333, $this->activationFunction->compute(-0.5));
    }

    public function test_differentiate()
    {
        $this->assertEquals(0.4444444444444444, $this->activationFunction->differentiate(1.0, 0.5));
        $this->assertEquals(0.5625, $this->activationFunction->differentiate(-0.5, -0.3333333333333333));
    }
}
