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
                [2.0, 1.0, -0.0049999, 0.0, 20.0, -0.0999999],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.0011999, 0.3100000, -0.0049000],
                [0.9900000, 0.0799999, -0.0002999],
                [0.0500000, -0.0051999, 0.5400000],
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
                [4.0, 2.0, 1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.0, 1.0, 1.0, 0.0099999, 0.0099999, 1.0, 0.0099999],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.0099999, 1.0, 0.0099999],
                [1.0, 1.0, 0.0099999],
                [1.0, 0.0099999, 1.0],
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

                [0.0010000, -0.0000100, 0.0000999, -0.0000009],
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

        static::assertEqualsWithDelta($expected, $activations, 1e-7);
    }

    #[Test]
    #[TestDox('Correctly handles boundary values during activation')]
    #[DataProvider('boundaryProvider')]
    public function testBoundaryActivate(NDArray $input, array $expected) : void
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
}
