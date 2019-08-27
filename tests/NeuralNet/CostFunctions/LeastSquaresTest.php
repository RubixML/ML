<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Rubix\Tensor\Tensor;
use Rubix\Tensor\Vector;
use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;
use Generator;

class LeastSquaresTest extends TestCase
{
    protected $costFn;

    public function setUp()
    {
        $this->costFn = new LeastSquares();
    }

    public function test_build_cost_function()
    {
        $this->assertInstanceOf(LeastSquares::class, $this->costFn);
        $this->assertInstanceOf(CostFunction::class, $this->costFn);
    }

    /**
     * @dataProvider compute_provider
     */
    public function test_compute(Tensor $output, Tensor $target, float $expected)
    {
        $loss = $this->costFn->compute($output, $target);

        $this->assertEquals($expected, $loss);
    }

    public function compute_provider() : Generator
    {
        yield [
            Matrix::quick([[0.99]]),
            Matrix::quick([[1.0]]),
            0.00010000000000000018,
        ];

        yield [
            Matrix::quick([[1000.]]),
            Matrix::quick([[1.]]),
            998001.0,
        ];

        yield [
            Matrix::quick([[33.98], [20.], [4.6], [44.2], [38.5]]),
            Matrix::quick([[36.], [22.], [18.], [41.5], [38.]]),
            39.036080000000005,
        ];
    }

    /**
     * @dataProvider differentiate_provider
     */
    public function test_differentiate(Tensor $output, Tensor $target, array $expected)
    {
        $gradient = $this->costFn->differentiate($output, $target)->asArray();

        $this->assertEquals($expected, $gradient);
    }

    public function differentiate_provider() : Generator
    {
        yield [
            Vector::quick([0.99]),
            Vector::quick([1.0]),
            [-0.010000000000000009],
        ];

        yield [
            Vector::quick([1000.]),
            Vector::quick([1.]),
            [999.0],
        ];

        yield [
            Matrix::quick([[33.98], [20.], [4.6], [44.2], [38.5]]),
            Matrix::quick([[36.], [22.], [18.], [41.5], [38.]]),
            [
                [-2.020000000000003],
                [-2.0],
                [-13.4],
                [2.700000000000003],
                [0.5],
            ],
        ];
    }
}
