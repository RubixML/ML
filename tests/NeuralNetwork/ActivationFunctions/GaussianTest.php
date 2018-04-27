<?php

use Rubix\Engine\NeuralNet\ActivationFunctions\Gaussian;
use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class GaussianTest extends TestCase
{
    protected $activationFunction;

    public function setUp()
    {
        $this->activationFunction = new Gaussian();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(Gaussian::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $this->assertEquals(0.36787944117144233, $this->activationFunction->compute(1.0));
        $this->assertEquals(0.7788007830714049, $this->activationFunction->compute(-0.5));
    }

    public function test_differentiate()
    {
        $this->assertEquals(-0.7357588823428847, $this->activationFunction->differentiate(1.0, 0.36787944117144233));
        $this->assertEquals(0.7788007830714049, $this->activationFunction->differentiate(-0.5, 0.7788007830714049));
    }
}
