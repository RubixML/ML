<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;

class LeastSquaresTest extends TestCase
{
    protected $costFn;

    protected $expected;

    protected $activation;

    public function setUp()
    {
        $this->expected = Matrix::quick([[36.], [22.], [18.], [41.5], [38.]]);

        $this->activation = Matrix::quick([[33.98], [20.], [4.6], [44.2], [38.5]]);

        $this->costFn = new LeastSquares();
    }

    public function test_build_cost_function()
    {
        $this->assertInstanceOf(LeastSquares::class, $this->costFn);
        $this->assertInstanceOf(CostFunction::class, $this->costFn);
    }

    public function test_compute()
    {
        $loss = $this->costFn->compute($this->expected, $this->activation)->asArray();

        $expected = [
            [4.0804000000000125],
            [4.0],
            [179.56],
            [7.290000000000015],
            [0.25],
        ];

        $this->assertEquals($expected, $loss);
    }

    public function test_differentiate()
    {
        $gradient = $this->costFn->differentiate($this->expected, $this->activation)->asArray();

        $expected = [
            [-2.020000000000003],
            [-2.0],
            [-13.4],
            [2.700000000000003],
            [0.5],
        ];

        $this->assertEquals($expected, $gradient);
    }
}
