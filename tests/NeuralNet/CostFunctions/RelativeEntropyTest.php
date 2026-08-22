<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use NumPower;
use NDArray;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('CostFunctions')]
#[CoversClass(RelativeEntropy::class)]
class RelativeEntropyTest extends TestCase
{
    protected RelativeEntropy $costFn;

    public static function computeProvider() : Generator
    {
        yield [
            NumPower::array([]),
            NumPower::array([]),
            NAN,
        ];

        yield [
            NumPower::array([
                [0.99, 0.01, 0.0],
            ]),
            NumPower::array([
                [1.0, 0.0, 0.0],
            ]),
            0.0033500,
        ];

        yield [
            NumPower::array([
                [0.2, 0.4, 0.4],
            ]),
            NumPower::array([
                [0.0, 1.0, 0.0],
            ]),
            0.3054301,
        ];

        yield [
            NumPower::array([
                [0.0, 0.1, 0.9],
            ]),
            NumPower::array([
                [1.0, 0.0, 0.0],
            ]),
            6.1402268,
        ];

        yield [
            NumPower::array([
                [0.2, 0.1, 0.7],
                [0.0, 0.9, 0.1],
                [0.1, 0.3, 0.6],
            ]),
            NumPower::array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            0.1080955,
        ];
    }

    public static function differentiateProvider() : Generator
    {
        yield [
            NumPower::array([
                [0.99, 0.01, 0.0],
            ]),
            NumPower::array([
                [1.0, 0.0, 0.0],
            ]),
            [
                [-0.0101010, 0.999999, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [0.2, 0.4, 0.4],
            ]),
            NumPower::array([
                [0.0, 1.0, 0.0],
            ]),
            [
                [0.9999999, -1.5, 0.9999999],
            ],
        ];

        yield [
            NumPower::array([
                [0.0, 0.1, 0.9],
            ]),
            NumPower::array([
                [1.0, 0.0, 0.0],
            ]),
            [
                [-100000000.0, 0.9999999, 0.9999999],
            ],
        ];

        yield [
            NumPower::array([
                [0.2, 0.1, 0.7],
                [0.0, 0.9, 0.1],
                [0.1, 0.3, 0.6],
            ]),
            NumPower::array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            [
                [0.9999999, 0.9999999, -0.4285714],
                [0.0, -0.1111111, 0.9999999],
                [0.9999999, 0.9999999, -0.6666666],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->costFn = new RelativeEntropy();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Relative Entropy', (string) $this->costFn);
    }

    #[Test]
    #[TestDox('Throws exception when output and target shapes do not match in compute')]
    public function testComputeThrowsExceptionOnShapeMismatch() : void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Output and target must have the same shape.');

        $output = NumPower::array([[1.0, 2.0, 3.0]]);
        $target = NumPower::array([[1.0, 2.0]]);

        $this->costFn->compute($output, $target);
    }

    #[Test]
    #[TestDox('Throws exception when output and target shapes do not match in differentiate')]
    public function testDifferentiateThrowsExceptionOnShapeMismatch() : void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Output and target must have the same shape.');

        $output = NumPower::array([[1.0, 2.0, 3.0]]);
        $target = NumPower::array([[1.0, 2.0]]);

        $this->costFn->differentiate($output, $target);
    }

    #[Test]
    #[TestDox('Compute loss score')]
    #[DataProvider('computeProvider')]
    public function testCompute(NDArray $output, NDArray $target, float $expected) : void
    {
        $loss = $this->costFn->compute($output, $target);

        if (is_nan($expected)) {
            self::assertNan($loss);
        } else {
            self::assertEqualsWithDelta($expected, $loss, 1e-7);
        }
    }

    #[Test]
    #[TestDox('Calculate gradient of cost function')]
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(NDArray $output, NDArray $target, array $expected) : void
    {
        $gradient = $this->costFn->differentiate($output, $target);

        $gradientArray = $gradient->toArray();

        self::assertEqualsWithDelta($expected, $gradientArray, 1e-7);
    }
}
