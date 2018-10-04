<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;

class RelativeEntropyTest extends TestCase
{
    protected $costFunction;

    protected $expected;

    protected $activation;

    protected $delta;

    public function setUp()
    {
        $this->expected = new Matrix([[1.], [0.], [0.], [1.], [0.]]);

        $this->activation = new Matrix([[0.99], [0.2], [0.7], [0.80], [0.02]]);

        $this->delta = new Matrix([
            [0.010050335852491413],
            [-4.444226394744681E-19],
            [-4.569502691594218E-19],
            [0.22314355128920965],
            [-4.2139678854452766E-19],
        ]);

        $this->costFunction = new RelativeEntropy();
    }

    public function test_build_cost_function()
    {
        $this->assertInstanceOf(RelativeEntropy::class, $this->costFunction);
        $this->assertInstanceOf(CostFunction::class, $this->costFunction);
    }

    public function test_compute()
    {
        $cost = $this->costFunction
            ->compute($this->expected, $this->activation)
            ->asArray();

        $this->assertEquals($this->delta->asArray(), $cost);
    }

    public function test_differentiate()
    {
        $derivative = $this->costFunction
            ->differentiate($this->expected, $this->activation, $this->delta)
            ->asArray();

        $outcome = [
            [-0.01010101010101011],
            [1.0],
            [1.0],
            [-0.24999999999999994],
            [1.0],
        ];

        $this->assertEquals($outcome, $derivative);
    }
}
