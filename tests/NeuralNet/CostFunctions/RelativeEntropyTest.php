<?php

namespace Rubix\Tests\NeuralNet\CostFunctions;

use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;

class RelativeEntropyTest extends TestCase
{
    const TOLERANCE = 1e-10;

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
            [-0.],
            [-0.],
            [0.22314355128920965],
            [-0.],
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
            [-0.010101009998979706],
            [1.],
            [1.],
            [-0.24999999984374993],
            [1.],
        ];

        $this->assertEquals($outcome, $derivative);
    }
}
