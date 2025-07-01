<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\SELU;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\SELU\SELU;

#[Group('ActivationFunctions')]
#[CoversClass(SELU::class)]
class SELUTest extends TestCase
{
    /**
     * @var SELU
     */
    protected SELU $activationFn;

    /**
     * The value at which leakage starts to saturate.
     *
     * @var float
     */
    public const ALPHA = 1.6732632;

    /**
     * The scaling coefficient.
     *
     * @var float
     */
    public const LAMBDA = 1.0507009;

    /**
     * The scaling coefficient multiplied by alpha.
     *
     * @var float
     */
    protected const BETA = self::LAMBDA * self::ALPHA;

    /**
     * @return Generator<array>
     */
    public static function computeProvider() : Generator
    {
        yield [
            NumPower::array([
                [2.0, 1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [
                    2.10140180,
                    1.05070090,
                    -0.6917580,
                    0.0,
                    21.0140190,
                    -1.7580193
                ],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [
                    self::BETA * (exp(-0.12) - 1.0),
                    0.31 * self::LAMBDA,
                    self::BETA * (exp(-0.49) - 1.0)
                ],
                [
                    0.99 * self::LAMBDA,
                    0.08 * self::LAMBDA,
                    self::BETA * (exp(-0.03) - 1.0)
                ],
                [
                    0.05 * self::LAMBDA,
                    self::BETA * (exp(-0.52) - 1.0),
                    0.54 * self::LAMBDA
                ],
            ],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function differentiateProvider() : Generator
    {
        yield [
            NumPower::array([
                [2.0, 1.0, -0.5, 0.0, 20.0, -10.0, -20],
            ]),
            [
                [
                    self::LAMBDA,
                    self::LAMBDA,
                    1.0663410,
                    1.7580991,
                    self::LAMBDA,
                    0.0000798,
                    0.0
                ],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [
                    self::BETA * exp(-0.12),
                    self::LAMBDA,
                    self::BETA * exp(-0.49)
                ],
                [
                    self::LAMBDA,
                    self::LAMBDA,
                    self::BETA * exp(-0.03)
                ],
                [
                    self::LAMBDA,
                    self::BETA * exp(-0.52),
                    self::LAMBDA
                ],
            ],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function zeroRegionProvider() : Generator
    {
        // Test exactly at zero
        yield [
            NumPower::array([[0.0]]),
            [[0.0]],
            [[1.7580991983413696]],
        ];

        // Test very small positive values
        yield [
            NumPower::array([[1e-15, 1e-10, 1e-7]]),
            [[1e-15 * self::LAMBDA, 1e-10 * self::LAMBDA, 1e-7 * self::LAMBDA]],
            [[self::LAMBDA, self::LAMBDA, self::LAMBDA]],
        ];

        // Test very small negative values
        yield [
            NumPower::array([[-1e-15, -1e-10, -1e-7]]),
            [
                [
                    self::BETA * (exp(-1e-15) - 1.0),
                    self::BETA * (exp(-1e-10) - 1.0),
                    self::BETA * (exp(-1e-7) - 1.0),
                ],
            ],
            [
                [
                    self::BETA * exp(-1e-15),
                    self::BETA * exp(-1e-10),
                    self::BETA * exp(-1e-7),
                ],
            ],
        ];

        // Test values around machine epsilon
        yield [
            NumPower::array([[PHP_FLOAT_EPSILON, -PHP_FLOAT_EPSILON]]),
            [
                [
                    PHP_FLOAT_EPSILON * self::LAMBDA,
                    self::BETA * (exp(-PHP_FLOAT_EPSILON) - 1.0),
                ],
            ],
            [
                [
                    self::LAMBDA,
                    self::BETA * exp(-PHP_FLOAT_EPSILON),
                ],
            ],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new SELU();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('SELU', (string) $this->activationFn);
    }

    #[Test]
    #[TestDox('Correctly activates the input')]
    #[DataProvider('computeProvider')]
    public function testActivate(NDArray $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();

        static::assertEqualsWithDelta($expected, $activations, 1e-7);
    }

    #[Test]
    #[TestDox('Correctly differentiates the input')]
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(NDArray $input, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input)->toArray();

        static::assertEqualsWithDelta($expected, $derivatives, 1e-7);
    }

    #[Test]
    #[TestDox('Correctly handles values around zero')]
    #[DataProvider('zeroRegionProvider')]
    public function testZeroRegion(NDArray $input, array $expectedActivation, array $expectedDerivative) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();
        $derivatives = $this->activationFn->differentiate($input)->toArray();

        static::assertEqualsWithDelta($expectedActivation, $activations, 1e-7);
        static::assertEqualsWithDelta($expectedDerivative, $derivatives, 1e-7);
    }
}
