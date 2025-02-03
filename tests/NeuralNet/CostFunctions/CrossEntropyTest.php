<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('CostFunctions')]
#[CoversClass(CrossEntropy::class)]
class CrossEntropyTest extends TestCase
{
    protected CrossEntropy $costFn;

    public static function computeProvider() : Generator
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

    protected function setUp() : void
    {
        $this->costFn = new CrossEntropy();
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

        $this->assertEqualsWithDelta($expected, $loss, 1e-8);
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

        $this->assertEqualsWithDelta($expected, $gradient, 1e-8);
    }
}
