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
                [2.0, 1.0, -0.05000000074505806, 0.0, 20.0, -1.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.012000000104308128, 0.3100000023841858, -0.049000002443790436],
                [0.9900000095367432, 0.07999999821186066, -0.003000000026077032],
                [0.05000000074505806, -0.05199999734759331, 0.5400000214576721],
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
                [1.0, 1.0, 0.10000000149011612, 0.10000000149011612, 1.0, 0.10000000149011612],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.10000000149011612, 1.0, 0.10000000149011612],
                [1.0, 1.0, 0.10000000149011612],
                [1.0, 0.10000000149011612, 1.0],
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
                [-10.0, -50.0, -100.0],
            ],
        ];

        // Test values close to zero
        yield [
            NumPower::array([
                [0.001, -0.001, 0.0001, -0.0001],
            ]),
            [

                [0.0010000000474974513, -0.00010000000474974513, 0.00009999999747378752, -0.000009999999747378752],
            ],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new LeakyReLU(0.1);
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
        static::assertEquals('Leaky ReLU (leakage: 0.1)', (string) $this->activationFn);
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
