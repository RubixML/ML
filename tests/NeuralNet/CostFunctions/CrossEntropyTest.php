<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group CostFunctions
 * @covers \Rubix\ML\NeuralNet\CostFunctions\CrossEntropy
 */
class CrossEntropyTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\CostFunctions\CrossEntropy
     */
    protected $costFn;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->costFn = new CrossEntropy();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(CrossEntropy::class, $this->costFn);
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
                [0.99, 0.01, 0.0],
            ]),
            Matrix::quick([
                [1.0, 0.0, 0.0],
            ]),
            0.00335011195116715,
        ];

        yield [
            Matrix::quick([
                [0.2, 0.4, 0.4],
            ]),
            Matrix::quick([
                [0.0, 1.0, 0.0],
            ]),
            0.3054302439580517,
        ];

        yield [
            Matrix::quick([
                [0.0, 0.1, 0.9],
            ]),
            Matrix::quick([
                [1.0, 0.0, 0.0],
            ]),
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
                [0.99, 0.01, 0.0],
            ]),
            Matrix::quick([
                [1.0, 0.0, 0.0],
            ]),
            [
                [-1.01010101010101, 1.01010101010101, 0.0],
            ],
        ];

        yield [
            Matrix::quick([
                [0.2, 0.4, 0.4],
            ]),
            Matrix::quick([
                [0.0, 1.0, 0.0],
            ]),
            [
                [1.2499999999999998, -2.5, 1.6666666666666667],
            ],
        ];

        yield [
            Matrix::quick([
                [0.0, 0.1, 0.9],
            ]),
            Matrix::quick([
                [1.0, 0.0, 0.0],
            ]),
            [
                [-100000000.0, 1.111111111111111, 10.000000000000002],
            ],
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
                [0.0, -1.1111111111111112, 1.111111111111111],
                [1.111111111111111, 1.4285714285714286, -1.6666666666666667],
            ],
        ];
    }
}
