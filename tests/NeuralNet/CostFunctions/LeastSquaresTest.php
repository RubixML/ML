<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('CostFunctions')]
#[CoversClass(LeastSquares::class)]
class LeastSquaresTest extends TestCase
{
    protected LeastSquares $costFn;

    public static function computeProvider() : Generator
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

    public static function differentiateProvider() : Generator
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

    protected function setUp() : void
    {
        $this->costFn = new LeastSquares();
    }

    /**
     * @param Matrix $output
     * @param Matrix $target
     * @param float $expected
     */
    #[DataProvider('computeProvider')]
    public function testCompute(Matrix $output, Matrix $target, float $expected) : void
    {
        $loss = $this->costFn->compute(output: $output, target: $target);

        $this->assertEquals($expected, $loss);
    }

    /**
     * @param Matrix $output
     * @param Matrix $target
     * @param list<list<float>> $expected
     */
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(Matrix $output, Matrix $target, array $expected) : void
    {
        $gradient = $this->costFn->differentiate(output: $output, target: $target)->asArray();

        $this->assertEquals($expected, $gradient);
    }
}
