<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\Softplus;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\Softplus\Softplus;

#[Group('ActivationFunctions')]
#[CoversClass(Softplus::class)]
class SoftplusTest extends TestCase
{
    /**
     * @var Softplus
     */
    protected Softplus $activationFn;

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
                [2.1269280, 1.3132617, 0.4740769, 0.6931472, 20.0000000, 0.0000454],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.6349461, 0.8601119, 0.4778640],
                [1.3059610, 0.7339470, 0.6782596],
                [0.7184596, 0.4665731, 0.9991626],
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
                [0.8807971, 0.7310586, 0.3775407, 0.5000000, 1.0000000, 0.0000454],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.4700359, 0.5768852, 0.3798935],
                [0.7290879, 0.5199893, 0.4925005],
                [0.5124973, 0.3728522, 0.6318124],
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
            [[0.6931472]],
            [[0.5000000]],
        ];

        // Test very small positive values
        yield [
            NumPower::array([[1e-15, 1e-10, 1e-7]]),
            [[0.6931471, 0.6931471, 0.6931471]],
            [[0.5000000, 0.5000000, 0.5000001]],
        ];

        // Test very small negative values
        yield [
            NumPower::array([[-1e-15, -1e-10, -1e-7]]),
            [[0.6931472, 0.6931472, 0.6931471]],
            [[0.5000000, 0.5000000, 0.5000000]],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function extremeValuesProvider() : Generator
    {
        // Test with large positive values
        yield [
            NumPower::array([[10.0, 20.0, 50.0]]),
            [[10.0000457, 20.0000000, 50.0000000]],
            [[0.9999546, 1.0000000, 1.0000000]],
        ];

        // Test with large negative values
        yield [
            NumPower::array([[-10.0, -20.0, -50.0]]),
            [[0.0000454, 0.0000000, 0.0000000]],
            [[0.0000454, 0.0000000, 0.0000000]],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new Softplus();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Soft Plus', (string) $this->activationFn);
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

    #[Test]
    #[TestDox('Correctly handles extreme values')]
    #[DataProvider('extremeValuesProvider')]
    public function testExtremeValues(NDArray $input, array $expectedActivation, array $expectedDerivative) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();
        $derivatives = $this->activationFn->differentiate($input)->toArray();

        static::assertEqualsWithDelta($expectedActivation, $activations, 1e-7);
        static::assertEqualsWithDelta($expectedDerivative, $derivatives, 1e-7);
    }
}
