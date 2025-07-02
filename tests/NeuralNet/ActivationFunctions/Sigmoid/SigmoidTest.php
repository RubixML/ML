<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\Sigmoid;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid\Sigmoid;

#[Group('ActivationFunctions')]
#[CoversClass(Sigmoid::class)]
class SigmoidTest extends TestCase
{
    /**
     * @var Sigmoid
     */
    protected Sigmoid $activationFn;

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
                [0.8807970, 0.7310586, 0.3775407, 0.5000000, 0.9999999, 0.0000454],
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
    public static function differentiateProvider() : Generator
    {
        yield [
            NumPower::array([
                [0.8807970, 0.7310586, 0.3775407, 0.5000000, 0.9999999, 0.0000454, 1.0, 0.2],
            ]),
            [
                [0.1049936, 0.1966119, 0.2350038, 0.2500000, 0.0000001, 0.0000454, 0.0, 0.16],
            ],
        ];

        yield [
            NumPower::array([
                [0.4700395, 0.5768852, 0.3799707],
                [0.7290795, 0.5199968, 0.4925041],
                [0.5124974, 0.3728375, 0.6319357],
            ]),
            [
                [0.2491023, 0.2440886, 0.2355929],
                [0.1975225, 0.2496001, 0.2499437],
                [0.2498438, 0.2338296, 0.2325929],
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
            [[0.5]],
            [[0.25]],
        ];

        // Test very small positive values
        yield [
            NumPower::array([[1e-15, 1e-10, 1e-7]]),
            [[0.5000000000000005, 0.5000000000000001, 0.5000000001]],
            [[0.25, 0.25, 0.25]],
        ];

        // Test very small negative values
        yield [
            NumPower::array([[-1e-15, -1e-10, -1e-7]]),
            [[0.4999999999999995, 0.4999999999999999, 0.4999999999]],
            [[0.25, 0.25, 0.25]],
        ];

        // Test values around machine epsilon
        yield [
            NumPower::array([[PHP_FLOAT_EPSILON, -PHP_FLOAT_EPSILON]]),
            [[0.5 + PHP_FLOAT_EPSILON/2, 0.5 - PHP_FLOAT_EPSILON/2]],
            [[0.25, 0.25]],
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
            [[0.9999546, 0.9999999, 1.0]],
            [[0.0000454, 0.0000001, 0.0]],
        ];

        // Test with large negative values
        yield [
            NumPower::array([[-10.0, -20.0, -50.0]]),
            [[0.0000454, 0.0000001, 0.0]],
            [[0.0000454, 0.0000001, 0.0]],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new Sigmoid();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Sigmoid', (string) $this->activationFn);
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
    #[TestDox('Correctly differentiates the output')]
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(NDArray $output, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($output)->toArray();

        static::assertEqualsWithDelta($expected, $derivatives, 1e-7);
    }

    #[Test]
    #[TestDox('Correctly handles values around zero')]
    #[DataProvider('zeroRegionProvider')]
    public function testZeroRegion(NDArray $input, array $expectedActivation, array $expectedDerivative) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();
        $derivatives = $this->activationFn->differentiate($this->activationFn->activate($input))->toArray();

        static::assertEqualsWithDelta($expectedActivation, $activations, 1e-7);
        static::assertEqualsWithDelta($expectedDerivative, $derivatives, 1e-7);
    }

    #[Test]
    #[TestDox('Correctly handles extreme values')]
    #[DataProvider('extremeValuesProvider')]
    public function testExtremeValues(NDArray $input, array $expectedActivation, array $expectedDerivative) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();
        $derivatives = $this->activationFn->differentiate($this->activationFn->activate($input))->toArray();

        static::assertEqualsWithDelta($expectedActivation, $activations, 1e-7);
        static::assertEqualsWithDelta($expectedDerivative, $derivatives, 1e-7);
    }
}
