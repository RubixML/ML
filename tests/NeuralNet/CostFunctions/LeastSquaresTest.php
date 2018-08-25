<?php

namespace Rubix\Tests\NeuralNet\CostFunctions;

use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;

class LeastSquaresTest extends TestCase
{
    protected $costFunction;

    protected $expected;

    protected $activation;

    protected $computed;

    public function setUp()
    {
        $this->expected = 1.0;

        $this->activation = 0.8;

        $this->computed = 0.01999999999999999;

        $this->costFunction = new LeastSquares();
    }

    public function test_build_cost_function()
    {
        $this->assertInstanceOf(LeastSquares::class, $this->costFunction);
        $this->assertInstanceOf(CostFunction::class, $this->costFunction);
    }

    public function test_compute()
    {
        $cost = $this->costFunction->compute($this->expected, $this->activation);

        $this->assertEquals($this->computed, $cost);
    }

    public function test_differentiate()
    {
        $derivative = $this->costFunction->differentiate($this->expected, $this->activation, $this->computed);

        $this->assertEquals(-0.19999999999999996, $derivative);
    }
}
