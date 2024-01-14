<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group CostFunctions
 * @covers \Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy
 */
class RelativeEntropyTest extends TestCase
{
    /**
     * @var RelativeEntropy
     */
    protected $costFn;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->costFn = new RelativeEntropy();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(RelativeEntropy::class, $this->costFn);
        $this->assertInstanceOf(CostFunction::class, $this->costFn);
    }

    /**
     * @test
     * @dataProvider computeProvider
     *
     * @param Matrix $output
     * @param Matrix $target
     * @param float $expected
     */
    public function compute(Matrix $output, Matrix $target, float $expected) : void
    {
        $loss = $this->costFn->compute($output, $target);

        $this->assertEqualsWithDelta($expected, $loss, 1e-8);
    }

    /**
     * @return \Generator<mixed[]>
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
            0.003350065899465309,
        ];

        yield [
            Matrix::quick([
                [0.2, 0.4, 0.4],
            ]),
            Matrix::quick([
                [0.0, 1.0, 0.0],
            ]),
            0.3054301295726089,
        ];

        yield [
            Matrix::quick([
                [0.0, 0.1, 0.9],
            ]),
            Matrix::quick([
                [1.0, 0.0, 0.0],
            ]),
            6.140226799872736,
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
            0.10809558439335247,
        ];
    }

    /**
     * @test
     * @dataProvider differentiateProvider
     *
     * @param Matrix $output
     * @param Matrix $target
     * @param list<list<float>> $expected
     */
    public function differentiate(Matrix $output, Matrix $target, array $expected) : void
    {
        $gradient = $this->costFn->differentiate($output, $target)->asArray();

        $this->assertEqualsWithDelta($expected, $gradient, 1e-8);
    }

    /**
     * @return \Generator<mixed[]>
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
                [-0.01010101010101011, 0.999999, 0.0],
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
                [0.9999999500000001, -1.4999999999999998, 0.999999975],
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
                [-99999999.0, 0.9999999, 0.9999999888888889],
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
                [0.9999999500000001, 0.9999999, -0.42857142857142866],
                [0.0, -0.11111111111111108, 0.9999999],
                [0.9999999, 0.9999999666666667, -0.6666666666666667],
            ],
        ];
    }
}
