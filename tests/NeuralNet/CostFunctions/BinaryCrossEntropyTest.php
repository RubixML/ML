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
            NumPower::array([]),
            NumPower::array([]),
            NAN,
        ];

        yield [
            NumPower::array([
                [0.99],
            ]),
            NumPower::array([
                [1.0],
            ]),
            0.0100503,
        ];

        yield [
            NumPower::array([
                [0.7],
            ]),
            NumPower::array([
                [1.0],
            ]),
            0.3566749,
        ];

        yield [
            NumPower::array([
                [0.01],
            ]),
            NumPower::array([
                [1.0],
            ]),
            4.6051702,
        ];

        yield [
            NumPower::array([
                [0.9],
                [0.1],
            ]),
            NumPower::array([
                [1.0],
                [0.0],
            ]),
            0.1053605,
        ];
    }

    public static function differentiateProvider() : Generator
    {
        yield [
            NumPower::array([
                [0.99],
            ]),
            NumPower::array([
                [1.0],
            ]),
            [
                [-1.0101010],
            ],
        ];

        yield [
            NumPower::array([
                [0.7],
            ]),
            NumPower::array([
                [1.0],
            ]),
            [
                [-1.4285714],
            ],
        ];

        yield [
            NumPower::array([
                [0.01],
            ]),
            NumPower::array([
                [1.0],
            ]),
            [
                [-100.0],
            ],
        ];

        yield [
            NumPower::array([
                [0.9],
                [0.1],
            ]),
            NumPower::array([
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

        $output = NumPower::array([[1.0, 2.0, 3.0]]);
        $target = NumPower::array([[1.0, 2.0]]);

        $this->costFn->compute($output, $target);
    }

    #[Test]
    #[TestDox('Throws exception when output and target shapes do not match in differentiate')]
    public function differentiateThrowsExceptionOnShapeMismatch() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $output = NumPower::array([[1.0, 2.0, 3.0]]);
        $target = NumPower::array([[1.0, 2.0]]);

        $this->costFn->differentiate($output, $target);
    }

    #[Test]
    #[TestDox('Compute loss score')]
    #[DataProvider('computeProvider')]
    public function compute(NDArray $output, NDArray $target, float $expected) : void
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
    public function differentiate(NDArray $output, NDArray $target, array $expected) : void
    {
        $gradient = $this->costFn->differentiate($output, $target);

        $gradientArray = $gradient->toArray();

        self::assertEqualsWithDelta($expected, $gradientArray, 1e-7);
    }
}
