<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group CostFunctions
 * @covers \Rubix\ML\NeuralNet\CostFunctions\LeastSquares
 */
class LeastSquaresTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\CostFunctions\LeastSquares
     */
    protected $costFn;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->costFn = new LeastSquares();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(LeastSquares::class, $this->costFn);
        $this->assertInstanceOf(CostFunction::class, $this->costFn);
    }

    /**
     * @test
     * @dataProvider computeProvider
     *
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @param float $expected
     */
    public function compute(Matrix $output, Matrix $target, float $expected) : void
    {
        $loss = $this->costFn->compute($output, $target);

        $this->assertEquals($expected, $loss);
    }

    /**
     * @return \Generator<array>
     */
    public function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [0.99],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            0.00010000000000000018,
        ];

        yield [
            Matrix::quick([
                [1000.0],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            998001.0,
        ];

        yield [
            Matrix::quick([
                [33.98],
                [20.0],
                [4.6],
                [44.2],
                [38.5],
            ]),
            Matrix::quick([
                [36.0],
                [22.0],
                [18.0],
                [41.5],
                [38.0],
            ]),
            39.036080000000005,
        ];
    }

    /**
     * @test
     * @dataProvider differentiateProvider
     *
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @param array[] $expected
     */
    public function differentiate(Matrix $output, Matrix $target, array $expected) : void
    {
        $gradient = $this->costFn->differentiate($output, $target)->asArray();

        $this->assertEquals($expected, $gradient);
    }

    /**
     * @return \Generator<array>
     */
    public function differentiateProvider() : Generator
    {
        yield [
            Matrix::quick([
                [0.99],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            [
                [-0.010000000000000009],
            ],
        ];

        yield [
            Matrix::quick([
                [1000.0],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            [
                [999.0],
            ],
        ];

        yield [
            Matrix::quick([
                [33.98],
                [20.0],
                [4.6],
                [44.2],
                [38.5],
            ]),
            Matrix::quick([
                [36.0],
                [22.0],
                [18.0],
                [41.5],
                [38.0],
            ]),
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
