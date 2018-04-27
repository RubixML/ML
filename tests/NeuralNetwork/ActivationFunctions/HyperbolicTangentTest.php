<?php

use Rubix\Engine\NeuralNet\ActivationFunctions\HyperbolicTangent;
use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class HyperbolicTangentTest extends TestCase
{
    protected $activationFunction;

    public function setUp()
    {
        $this->activationFunction = new HyperbolicTangent();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(HyperbolicTangent::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_compute()
    {
        $this->assertEquals(0.7615941559557649, $this->activationFunction->compute(1.0));
        $this->assertEquals(-0.46211715726000974, $this->activationFunction->compute(-0.5));
    }

    public function test_differentiate()
    {
        $this->assertEquals(0.41997434161402614, $this->activationFunction->differentiate(1.0, 0.7615941559557649));
        $this->assertEquals(0.7864477329659274, $this->activationFunction->differentiate(-0.5, -0.46211715726000974));
    }
}
