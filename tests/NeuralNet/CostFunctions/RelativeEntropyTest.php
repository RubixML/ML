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
        $this->expected = Matrix::quick([[1.], [0.], [0.], [1.], [0.]]);

        $this->activation = Matrix::quick([[0.99], [0.2], [0.7], [0.80], [0.02]]);

        $this->delta = Matrix::quick([
            [0.010050335852491413],
            [-1.6811242831518263E-7],
            [-1.8064005800013633E-7],
            [0.22314355128920965],
            [-1.4508657738524219E-7],
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
            [0.9999999500000001],
            [0.9999999857142856],
            [-0.24999999999999994],
            [0.9999994999999999],
        ];

        $this->assertEquals($outcome, $derivative);
    }
}
