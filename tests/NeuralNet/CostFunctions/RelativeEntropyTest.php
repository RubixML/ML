<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('CostFunctions')]
#[CoversClass(RelativeEntropy::class)]
class RelativeEntropyTest extends TestCase
{
    /**
     * @var RelativeEntropy
     */
    protected $costFn;

    /**
     * @return Generator<mixed[]>
     */
    public static function computeProvider() : Generator
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
     * @return Generator<mixed[]>
     */
    public static function differentiateProvider() : Generator
    {
        yield [
            Matrix::quick([
                [0.99, 0.01, 0.0],
            ]),
            Matrix::quick([
                [1.0, 0.0, 0.0],
            ]),
            [
                [-1.0101010101010102, -1.0e-6, -1.0],
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
                [-5.0e-8, -2.5, -2.5e-8],
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
                [-100000000.0, -1.0e-7, -1.1111111111111112e-8],
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
                [-5.0e-8, -1.0e-7, -1.4285714285714286],
                [-1.0, -1.1111111111111112, -1.0e-7],
                [-1.0e-7, -3.3333333333333334e-8, -1.6666666666666667],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->costFn = new RelativeEntropy();
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(RelativeEntropy::class, $this->costFn);
        $this->assertInstanceOf(CostFunction::class, $this->costFn);
    }

    /**
     * @param Matrix $output
     * @param Matrix $target
     * @param float $expected
     */
    #[DataProvider('computeProvider')]
    #[Test]
    public function compute(Matrix $output, Matrix $target, float $expected) : void
    {
        $loss = $this->costFn->compute($output, $target);

        $this->assertEqualsWithDelta($expected, $loss, 1e-8);
    }

    /**
     * @param Matrix $output
     * @param Matrix $target
     * @param list<list<float>> $expected
     */
    #[DataProvider('differentiateProvider')]
    #[Test]
    public function differentiate(Matrix $output, Matrix $target, array $expected) : void
    {
        $gradient = $this->costFn->differentiate($output, $target)->asArray();

        $this->assertEqualsWithDelta($expected, $gradient, 1e-8);
    }
}
