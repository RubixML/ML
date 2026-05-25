<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\CostFunctions\CrossEntropy;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use NumPower;
use NDArray;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy\CrossEntropy;
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
            0.0033501,
        ];

        yield [
            NumPower::array([
                [0.2, 0.4, 0.4],
            ]),
            NumPower::array([
                [0.0, 1.0, 0.0],
            ]),
            0.3054302,
        ];

        yield [
            NumPower::array([
                [0.0, 0.1, 0.9],
            ]),
            NumPower::array([
                [1.0, 0.0, 0.0],
            ]),
            6.1402269,
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
            0.1080956,
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
                [-1.0101009, 1.0101009, 0.0],
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
                [1.2499999, -2.5, 1.6666666],
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
                [-100000000.0, 1.1111111, 9.9999981],
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
                [1.2499999, 1.1111111, -1.4285714],
                [0.0, -1.1111111, 1.1111111],
                [1.1111111, 1.4285714, -1.6666666],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->costFn = new CrossEntropy();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Cross Entropy', (string) $this->costFn);
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
