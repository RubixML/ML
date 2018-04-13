<?php

use Rubix\Engine\NeuralNetwork\ActivationFunctions\Sigmoid;
use Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class SigmoidTest extends TestCase
{
    protected $activationFunction;

    public function setUp()
    {
        $this->activationFunction = new Sigmoid();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(Sigmoid::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $this->assertEquals(0.7310585786300049, $this->activationFunction->compute(1.0));
        $this->assertEquals(0.3775406687981454, $this->activationFunction->compute(-0.5));
    }

    public function test_differentiate()
    {
        $this->assertEquals(0.19661193324148185, $this->activationFunction->differentiate(1.0, 0.7310585786300049));
        $this->assertEquals(0.2350037122015945, $this->activationFunction->differentiate(-0.5, 0.3775406687981454));
    }
}
