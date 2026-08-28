<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use InvalidArgumentException;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('CostFunctions')]
#[CoversClass(LeastSquares::class)]
class LeastSquaresTest extends TestCase
{
    /**
     * @var LeastSquares
     */
    protected LeastSquares $costFn;

    /**
     * @return Generator<array>
     */
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
            0.0001000,
        ];

        yield [
            NumPower::array([
                [1000.0],
            ]),
            NumPower::array([
                [1.0],
            ]),
            998001.0,
        ];

        yield [
            NumPower::array([
                [33.98],
                [20.0],
                [4.6],
                [44.2],
                [38.5],
            ]),
            NumPower::array([
                [36.0],
                [22.0],
                [18.0],
                [41.5],
                [38.0],
            ]),
            39.0360776,
        ];
    }

    /**
     * @return Generator<array>
     */
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
                [-0.0099999],
            ],
        ];

        yield [
            NumPower::array([
                [1000.0],
            ]),
            NumPower::array([
                [1.0],
            ]),
            [
                [999.0],
            ],
        ];

        yield [
            NumPower::array([
                [33.98],
                [20.0],
                [4.6],
                [44.2],
                [38.5],
            ]),
            NumPower::array([
                [36.0],
                [22.0],
                [18.0],
                [41.5],
                [38.0],
            ]),
            [
                [-2.0200004],
                [-2.0],
                [-13.3999996],
                [2.7000007],
                [0.5],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->costFn = new LeastSquares();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Least Squares', (string) $this->costFn);
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

        // Convert NDArray to PHP array for comparison
        $gradientArray = $gradient->toArray();

        self::assertEqualsWithDelta($expected, $gradientArray, 1e-7);
    }
}
