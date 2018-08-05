<?php

namespace Rubix\Tests\NeuralNet\CostFunctions;

use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use Rubix\ML\NeuralNet\CostFunctions\Quadratic;
use PHPUnit\Framework\TestCase;

class QuadraticTest extends TestCase
{
    protected $costFunction;

    protected $expected;

    protected $activation;

    public function setUp()
    {
        $this->expected = 1.0;

        $this->activation = 0.8;

        $this->costFunction = new Quadratic();
    }

    public function test_build_cost_function()
    {
        $this->assertInstanceOf(Quadratic::class, $this->costFunction);
        $this->assertInstanceOf(CostFunction::class, $this->costFunction);
    }

    public function test_compute()
    {
        $cost = $this->costFunction->compute($this->expected, $this->activation);

        $this->assertEquals(0.01999999999999999, $cost);
    }

    public function test_differentiate()
    {
        $derivative = $this->costFunction->differentiate($this->expected, $this->activation);

        $this->assertEquals(0.19999999999999996, $derivative);
    }
}
