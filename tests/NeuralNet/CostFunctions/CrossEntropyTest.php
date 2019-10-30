<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Tensor\Tensor;
use Tensor\Vector;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;
use Generator;

class CrossEntropyTest extends TestCase
{
    protected $costFn;

    public function setUp()
    {
        $this->costFn = new CrossEntropy();
    }

    public function test_build_cost_function()
    {
        $this->assertInstanceOf(CrossEntropy::class, $this->costFn);
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
            Matrix::quick([[0.99, 0.01, 0.]]),
            Matrix::quick([[1., 0., 0.]]),
            0.00335011195116715,
        ];

        yield [
            Matrix::quick([[0.2, 0.4, 0.4]]),
            Matrix::quick([[0., 1., 0.]]),
            0.3054302439580517,
        ];

        yield [
            Matrix::quick([[0.0, 0.1, 0.9]]),
            Matrix::quick([[1., 0., 0.]]),
            6.140226914650789,
        ];

        yield [
            Matrix::quick([
                [0.2, 0.1, 0.7],
                [0.0, 0.9, 0.1],
                [0.1, 0.3, 0.6],
            ]),
            Matrix::quick([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            0.10809567592917217,
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
            Vector::quick([0.99, 0.01, 0.]),
            Vector::quick([1., 0., 0.]),
            [-1.01010101010101, 1.01010101010101, 0.],
        ];

        yield [
            Vector::quick([0.2, 0.4, 0.4]),
            Vector::quick([0., 1., 0.]),
            [1.2499999999999998, -2.5, 1.6666666666666667],
        ];

        yield [
            Vector::quick([0.0, 0.1, 0.9]),
            Vector::quick([1., 0., 0.]),
            [-100000000.0, 1.111111111111111, 10.000000000000002],
        ];

        yield [
            Matrix::quick([
                [0.2, 0.1, 0.7],
                [0.0, 0.9, 0.1],
                [0.1, 0.3, 0.6],
            ]),
            Matrix::quick([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            [
                [1.2499999999999998, 1.111111111111111, -1.4285714285714286],
                [0., -1.1111111111111112, 1.111111111111111],
                [1.111111111111111, 1.4285714285714286, -1.6666666666666667],
            ],
        ];
    }
}
