<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('CostFunctions')]
#[CoversClass(MulticlassCrossEntropy::class)]
class MulticlassCrossEntropyTest extends TestCase
{
    protected MulticlassCrossEntropy $costFn;

    public static function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [0.99, 0.01, 0.0],
            ]),
            Matrix::quick([
                [1.0, 0.0, 0.0],
            ]),
            0.0033501,
        ];

        yield [
            Matrix::quick([
                [0.2, 0.4, 0.4],
            ]),
            Matrix::quick([
                [0.0, 1.0, 0.0],
            ]),
            0.3054302,
        ];

        yield [
            Matrix::quick([
                [0.0, 0.1, 0.9],
            ]),
            Matrix::quick([
                [1.0, 0.0, 0.0],
            ]),
            6.1402269,
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
            0.1080956,
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
                [-1.0101010, 0.0, 0.0],
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
                [0.0, -2.5, 0.0],
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
                [-100000000.0, 0.0, 0.0],
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
                [0.0, 0.0, -1.4285714],
                [0.0, -1.1111111, 0.0],
                [0.0, 0.0, -1.6666666],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->costFn = new MulticlassCrossEntropy();
    }

    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Multiclass Cross Entropy', (string) $this->costFn);
    }

    #[Test]
    #[TestDox('Throws exception when output and target shapes do not match in compute')]
    public function computeThrowsExceptionOnShapeMismatch() : void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Output and target must have the same shape.');

        $output = Matrix::quick([[1.0, 2.0, 3.0]]);
        $target = Matrix::quick([[1.0, 2.0]]);

        $this->costFn->compute($output, $target);
    }

    #[Test]
    #[TestDox('Throws exception when output and target shapes do not match in differentiate')]
    public function differentiateThrowsExceptionOnShapeMismatch() : void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Output and target must have the same shape.');

        $output = Matrix::quick([[1.0, 2.0, 3.0]]);
        $target = Matrix::quick([[1.0, 2.0]]);

        $this->costFn->differentiate($output, $target);
    }

    #[Test]
    #[TestDox('Compute loss score')]
    #[DataProvider('computeProvider')]
    public function compute(Matrix $output, Matrix $target, float $expected) : void
    {
        $loss = $this->costFn->compute($output, $target);

        self::assertEqualsWithDelta($expected, $loss, 1e-7);
    }

    #[Test]
    #[TestDox('Calculate gradient of cost function')]
    #[DataProvider('differentiateProvider')]
    public function differentiate(Matrix $output, Matrix $target, array $expected) : void
    {
        $gradient = $this->costFn->differentiate($output, $target);

        $gradientArray = $gradient->asArray();

        self::assertEqualsWithDelta($expected, $gradientArray, 1e-7);
    }
}
