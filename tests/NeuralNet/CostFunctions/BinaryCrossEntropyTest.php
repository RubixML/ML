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
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('CostFunctions')]
#[CoversClass(BinaryCrossEntropy::class)]
class BinaryCrossEntropyTest extends TestCase
{
    protected BinaryCrossEntropy $costFn;

    public static function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [0.99],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            0.0100503,
        ];

        yield [
            Matrix::quick([
                [0.7],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            0.3566749,
        ];

        yield [
            Matrix::quick([
                [0.01],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            4.6051702,
        ];

        yield [
            Matrix::quick([
                [0.9],
                [0.1],
            ]),
            Matrix::quick([
                [1.0],
                [0.0],
            ]),
            0.1053605,
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
                [-1.0101010],
            ],
        ];

        yield [
            Matrix::quick([
                [0.7],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            [
                [-1.4285714],
            ],
        ];

        yield [
            Matrix::quick([
                [0.01],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            [
                [-100.0],
            ],
        ];

        yield [
            Matrix::quick([
                [0.9],
                [0.1],
            ]),
            Matrix::quick([
                [1.0],
                [0.0],
            ]),
            [
                [-1.1111111],
                [1.1111111],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->costFn = new BinaryCrossEntropy();
    }

    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Binary Cross Entropy', (string) $this->costFn);
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
