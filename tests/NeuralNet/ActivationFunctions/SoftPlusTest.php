<?php

use Rubix\Engine\NeuralNet\ActivationFunctions\SoftPlus;
use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class SoftPlusTest extends TestCase
{
    protected $activationFunction;

    public function setUp()
    {
        $this->activationFunction = new SoftPlus();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(SoftPlus::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $this->assertEquals(1.3132616875182228, $this->activationFunction->compute(1.0));
        $this->assertEquals(0.4740769841801067, $this->activationFunction->compute(-0.5));
    }

    public function test_differentiate()
    {
        $this->assertEquals(0.7880584423829144, $this->activationFunction->differentiate(1.0, 1.3132616875182228));
        $this->assertEquals(0.6163482688094494, $this->activationFunction->differentiate(-0.5, 0.4740769841801067));
    }
}
