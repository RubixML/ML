<?php

namespace Rubix\Tests\NeuralNet\CostFunctions;

use Rubix\ML\NeuralNet\CostFunctions\Exponential;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;

class ExponentialTest extends TestCase
{
    const TOLERANCE = 1e-10;

    protected $costFunction;

    protected $expected;

    protected $activation;

    protected $computed;

    public function setUp()
    {
        $this->expected = 1.0;

        $this->activation = 0.8;

        $this->computed = 1.0408107741923882;

        $this->costFunction = new Exponential(1.0);
    }

    public function test_build_cost_function()
    {
        $this->assertInstanceOf(Exponential::class, $this->costFunction);
        $this->assertInstanceOf(CostFunction::class, $this->costFunction);
    }

    public function test_compute()
    {
        $cost = $this->costFunction->compute($this->expected, $this->activation);

        $this->assertEquals($this->computed, $cost, '', self::TOLERANCE);
    }

    public function test_differentiate()
    {
        $derivative = $this->costFunction->differentiate($this->expected, $this->activation, $this->computed);

        $this->assertEquals(-0.4163243096769552, $derivative, '', self::TOLERANCE);
    }
}
