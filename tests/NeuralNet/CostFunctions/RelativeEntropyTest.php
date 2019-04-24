<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;

class RelativeEntropyTest extends TestCase
{
    protected $costFn;

    protected $expected;

    protected $activation;

    public function setUp()
    {
        $this->expected = Matrix::quick([[1.], [0.], [0.], [1.], [0.]]);

        $this->activation = Matrix::quick([[0.99], [0.2], [0.7], [0.80], [0.02]]);

        $this->costFn = new RelativeEntropy();
    }

    public function test_build_cost_function()
    {
        $this->assertInstanceOf(RelativeEntropy::class, $this->costFn);
        $this->assertInstanceOf(CostFunction::class, $this->costFn);
    }

    public function test_compute()
    {
        $cost = $this->costFn->compute($this->expected, $this->activation);

        $this->assertEquals(0.04663867866572951, $cost);
    }

    public function test_differentiate()
    {
        $derivative = $this->costFn
            ->differentiate($this->expected, $this->activation)
            ->asArray();

        $expected = [
            [-0.01010101010101011],
            [0.9999999500000001],
            [0.9999999857142856],
            [-0.24999999999999994],
            [0.9999994999999999],
        ];

        $this->assertEquals($expected, $derivative);
    }
}
