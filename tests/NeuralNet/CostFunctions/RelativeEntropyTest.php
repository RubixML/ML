<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Rubix\Tensor\Tensor;
use Rubix\Tensor\Vector;
use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;
use Generator;

class RelativeEntropyTest extends TestCase
{
    protected $costFn;

    public function setUp()
    {
        $this->costFn = new RelativeEntropy();
    }

    public function test_build_cost_function()
    {
        $this->assertInstanceOf(RelativeEntropy::class, $this->costFn);
        $this->assertInstanceOf(CostFunction::class, $this->costFn);
    }

    /**
     * @dataProvider compute_provider
     */
    public function test_compute(Tensor $output, Tensor $target, array $expected)
    {
        $loss = $this->costFn->compute($output, $target)->asArray();

        $this->assertEquals($expected, $loss);
    }

    public function compute_provider() : Generator
    {
        yield [
            Vector::quick([0.99, 0.01, 0.]),
            Vector::quick([1., 0., 0.]),
            [0.010050335853501506, -1.3815510557964274E-7, 0.],
        ];

        yield [
            Vector::quick([0.2, 0.4, 0.4]),
            Vector::quick([0., 1., 0.]),
            [-1.6811242831518263E-7, 0.9162907318741551, -1.750439001207821E-7],
        ];

        yield [
            Vector::quick([0.0, 0.1, 0.9]),
            Vector::quick([1., 0., 0.]),
            [18.420680743952367, -1.6118095650958322E-7, -1.831532022829454E-7],
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
                [-1.6811242831518263E-7, -1.6118095650958322E-7, 0.3566749439387324],
                [0., 0.10536051565782635, -1.6118095650958322E-7],
                [-1.6118095650958322E-7, -1.7216707939626428E-7, 0.5108256237659907],
            ],
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
            [-0.01010101010101011, 0.999999, 0.],
        ];

        yield [
            Vector::quick([0.2, 0.4, 0.4]),
            Vector::quick([0., 1., 0.]),
            [0.9999999500000001, -1.4999999999999998, 0.999999975],
        ];

        yield [
            Vector::quick([0.0, 0.1, 0.9]),
            Vector::quick([1., 0., 0.]),
            [-99999999.0, 0.9999999, 0.9999999888888889],
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
                [0.9999999500000001, 0.9999999, -0.42857142857142866],
                [0., -0.11111111111111108, 0.9999999],
                [0.9999999, 0.9999999666666667, -0.6666666666666667],
            ],
        ];
    }
}
