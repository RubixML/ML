<?php

use Rubix\Engine\NeuralNet\ActivationFunctions\ELU;
use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class ELUTest extends TestCase
{
    protected $activationFunction;

    public function setUp()
    {
        $this->activationFunction = new ELU(1.0);
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(ELU::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $this->assertEquals(1.0, $this->activationFunction->compute(1.0));
        $this->assertEquals(-0.3934693402873666, $this->activationFunction->compute(-0.5));
    }

    public function test_differentiate()
    {
        $this->assertEquals(1.0, $this->activationFunction->differentiate(1.0, 1.0));
        $this->assertEquals(0.6065306597126334, $this->activationFunction->differentiate(-0.5, -0.3934693402873666));
    }
}
