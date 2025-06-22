<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\LeakyReLU;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU\LeakyReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU\Exceptions\InvalidLeakageException;

#[Group('ActivationFunctions')]
#[CoversClass(LeakyReLU::class)]
class LeakyReLUTest extends TestCase
{
    /**
     * @var LeakyReLU
     */
    protected LeakyReLU $activationFn;

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
                [2.0, 1.0, -0.004999999888241291, 0.0, 20.0, -0.09999999403953552],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.0011999999405816197, 0.3100000023841858, -0.004900000058114529],
                [0.9900000095367432, 0.07999999821186066, -0.00029999998514540493],
                [0.05000000074505806, -0.005199999548494816, 0.5400000214576721],
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
                [2.0, 1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.0, 1.0, 0.009999999776482582, 0.009999999776482582, 1.0, 0.009999999776482582],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.009999999776482582, 1.0, 0.009999999776482582],
                [1.0, 1.0, 0.009999999776482582],
                [1.0, 0.009999999776482582, 1.0],
            ],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function boundaryProvider() : Generator
    {
        // Test very large positive values (should be equal to input)
        yield [
            NumPower::array([
                [100.0, 500.0, 1000.0],
            ]),
            [
                [100.0, 500.0, 1000.0],
            ],
        ];

        // Test very large negative values (should be input * leakage)
        yield [
            NumPower::array([
                [-100.0, -500.0, -1000.0],
            ]),
            [
                [-1.0, -5.0, -10.0],
            ],
        ];

        // Test values close to zero
        yield [
            NumPower::array([
                [0.001, -0.001, 0.0001, -0.0001],
            ]),
            [

                [0.0010000000474974513, -0.000010000000656873453, 0.00009999999747378752, -0.0000009999999974752427],
            ],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new LeakyReLU(0.01);
    }

    #[Test]
    #[TestDox('Can be constructed with valid leakage parameter')]
    public function testConstructorWithValidLeakage() : void
    {
        $activationFn = new LeakyReLU(0.2);

        static::assertInstanceOf(LeakyReLU::class, $activationFn);
        static::assertEquals('Leaky ReLU (leakage: 0.2)', (string) $activationFn);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with invalid leakage parameter')]
    public function testConstructorWithInvalidLeakage() : void
    {
        $this->expectException(InvalidLeakageException::class);

        new LeakyReLU(1.5);
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Leaky ReLU (leakage: 0.01)', (string) $this->activationFn);
    }

    #[Test]
    #[TestDox('Correctly activates the input')]
    #[DataProvider('computeProvider')]
    public function testActivate(NDArray $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();

        static::assertEqualsWithDelta($expected, $activations, 1e-16);
    }

    #[Test]
    #[TestDox('Correctly differentiates the input')]
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(NDArray $input, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input)->toArray();

        static::assertEqualsWithDelta($expected, $derivatives, 1e-16);
    }

    #[Test]
    #[TestDox('Correctly handles boundary values during activation')]
    #[DataProvider('boundaryProvider')]
    public function testBoundaryActivate(NDArray $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();

        static::assertEqualsWithDelta($expected, $activations, 1e-16);
    }
}
