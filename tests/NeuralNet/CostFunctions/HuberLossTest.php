<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use NDArray;
use NumPower;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\InvalidAlphaException;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('CostFunctions')]
#[CoversClass(HuberLoss::class)]
class HuberLossTest extends TestCase
{
    protected HuberLoss $costFn;

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
            0.0000499,
        ];

        yield [
            NumPower::array([
                [1000.0],
            ]),
            NumPower::array([
                [1.0],
            ]),
            998.0004882,
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
            3.3849148,
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
                [-0.0099995],
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
                [0.9999995],
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
                [-0.8961948],
                [-0.8944271],
                [-0.9972270],
                [0.9377487],
                [0.4472135],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->costFn = new HuberLoss(1.0);
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Huber Loss (alpha: 1)', (string) $this->costFn);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with invalid alpha parameter')]
    public function testConstructorWithInvalidAlpha() : void
    {
        $this->expectException(InvalidAlphaException::class);

        new HuberLoss(-1);
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

    /**
     * @test
     */
    public function differentiateMatchesNumericGradient() : void
    {
        $costFn = new HuberLoss(0.5);

        $output = Matrix::quick([
            [0.1, 0.5, 1.0],
            [2.0, 5.0, 10.0],
        ]);

        $target = Matrix::quick([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);

        $epsilon = 1e-6;

        $numeric = [];

        foreach ($output->asArray() as $i => $row) {
            foreach ($row as $j => $v) {
                $plus = $costFn->_compute($target[$i][$j] - ($v + $epsilon));
                $minus = $costFn->_compute($target[$i][$j] - ($v - $epsilon));

                $numeric[$i][$j] = ($plus - $minus) / (2.0 * $epsilon);
            }
        }

        $this->assertEqualsWithDelta($numeric, $costFn->differentiate($output, $target)->asArray(), 1e-8);
    }
}
