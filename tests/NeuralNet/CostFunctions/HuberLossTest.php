<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Tensor\Tensor;
use Tensor\Vector;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;
use Generator;

class HuberLossTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\CostFunctions\HuberLoss
     */
    protected $costFn;
    
    public function setUp() : void
    {
        $this->costFn = new HuberLoss(1.);
    }

    public function test_build_cost_function() : void
    {
        $this->assertInstanceOf(HuberLoss::class, $this->costFn);
        $this->assertInstanceOf(CostFunction::class, $this->costFn);
    }

    /**
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @param float $expected
     *
     * @dataProvider compute_provider
     */
    public function test_compute(Matrix $output, Matrix $target, float $expected) : void
    {
        $loss = $this->costFn->compute($output, $target);

        $this->assertEquals($expected, $loss);
    }

    public function compute_provider() : Generator
    {
        yield [
            Matrix::quick([[0.99]]),
            Matrix::quick([[1.0]]),
            4.9998750062396624E-5,
        ];

        yield [
            Matrix::quick([[1000.]]),
            Matrix::quick([[1.]]),
            998.0005005003751,
        ];

        yield [
            Matrix::quick([[33.98], [20.], [4.6], [44.2], [38.5]]),
            Matrix::quick([[36.], [22.], [18.], [41.5], [38.]]),
            3.384914773928223,
        ];
    }

    /**
     * @param \Tensor\Tensor $output
     * @param \Tensor\Tensor $target
     * @param array[] $expected
     *
     * @dataProvider differentiate_provider
     */
    public function test_differentiate(Tensor $output, Tensor $target, array $expected) : void
    {
        $gradient = $this->costFn->differentiate($output, $target)->asArray();

        $this->assertEquals($expected, $gradient);
    }

    /**
     * @return \Generator
     */
    public function differentiate_provider() : Generator
    {
        yield [
            Vector::quick([0.99]),
            Vector::quick([1.0]),
            [-0.009999500037496884],
        ];

        yield [
            Vector::quick([1000.]),
            Vector::quick([1.]),
            [0.999999498998874],
        ];

        yield [
            Matrix::quick([[33.98], [20.], [4.6], [44.2], [38.5]]),
            Matrix::quick([[36.], [22.], [18.], [41.5], [38.]]),
            [
                [-0.8961947919452747],
                [-0.8944271909999159],
                [-0.9972269926097788],
                [0.9377487607237037],
                [0.4472135954999579],
            ],
        ];
    }
}
