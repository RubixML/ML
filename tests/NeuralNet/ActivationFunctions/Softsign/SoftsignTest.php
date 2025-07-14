<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\Softsign;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\Softsign\Softsign;

#[Group('ActivationFunctions')]
#[CoversClass(Softsign::class)]
class SoftsignTest extends TestCase
{
    /**
     * @var Softsign
     */
    protected Softsign $activationFn;

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
                [0.6666667, 0.5000000, -0.3333333, 0.0000000, 0.9523810, -0.9090909],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.1071429, 0.2366412, -0.3288591],
                [0.4974874, 0.0740741, -0.0291262],
                [0.0476190, -0.3421053, 0.3506494],
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
                [0.1111111, 0.2500000, 0.4444444, 1.0000000, 0.0022676, 0.0082645],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.7971938, 0.5827166, 0.4504301],
                [0.2525188, 0.8573387, 0.9425959],
                [0.9070296, 0.4328254, 0.4216562],
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
            [[1.0]],
        ];

        // Test very small values
        yield [
            NumPower::array([[0.0000001, -0.0000001]]),
            [[0.000000099999999, -0.000000099999999]],
            [[0.9999998, 0.9999998]],
        ];

        // Test values around machine epsilon
        yield [
            NumPower::array([[PHP_FLOAT_EPSILON, -PHP_FLOAT_EPSILON]]),
            [[PHP_FLOAT_EPSILON / (1 + PHP_FLOAT_EPSILON), -PHP_FLOAT_EPSILON / (1 + PHP_FLOAT_EPSILON)]],
            [[1 / (1 + PHP_FLOAT_EPSILON) ** 2, 1 / (1 + PHP_FLOAT_EPSILON) ** 2]],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function extremeValuesProvider() : Generator
    {
        // Test with large positive values
        yield [
            NumPower::array([[10.0, 100.0, 1000.0]]),
            [[0.9090909, 0.9900990, 0.9990010]],
            [[0.00826446, 0.0000980, 0.0000009]],
        ];

        // Test with large negative values
        yield [
            NumPower::array([[-10.0, -100.0, -1000.0]]),
            [[-0.9090909, -0.9900990, -0.9990010]],
            [[0.00826446, 0.0000980, 0.0000009]],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new Softsign();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Softsign', (string) $this->activationFn);
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
