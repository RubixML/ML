<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\ThresholdedReLU;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU\ThresholdedReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU\Exceptions\InvalidThresholdException;

#[Group('ActivationFunctions')]
#[CoversClass(ThresholdedReLU::class)]
class ThresholdedReLUTest extends TestCase
{
    /**
     * @var ThresholdedReLU
     */
    protected ThresholdedReLU $activationFn;

    /**
     * @var float
     */
    protected float $threshold = 1.0;

    /**
     * @return Generator<array>
     */
    public static function computeProvider() : Generator
    {
        yield [
            NumPower::array([
                [2.0, 1.0, 0.5, 0.0, -1.0, 1.5, -0.5],
            ]),
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [1.2, 0.31, 1.49],
                [0.99, 1.08, 0.03],
                [1.05, 0.52, 1.54],
            ]),
            [
                [1.2, 0.0, 1.49],
                [0.0, 1.08, 0.0],
                [1.05, 0.0, 1.54],
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
                [2.0, 1.0, 0.5, 0.0, -1.0, 1.5, -0.5],
            ]),
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [1.2, 0.31, 1.49],
                [0.99, 1.08, 0.03],
                [1.05, 0.52, 1.54],
            ]),
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function thresholdValuesProvider() : Generator
    {
        yield [
            0.5,
            NumPower::array([
                [2.0, 1.0, 0.5, 0.0, -1.0],
            ]),
            [
                [2.0, 1.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
            ],
        ];

        yield [
            2.0,
            NumPower::array([
                [2.0, 1.0, 3.0, 0.0, 2.5],
            ]),
            [
                [0.0, 0.0, 3.0, 0.0, 2.5],
            ],
            [
                [0.0, 0.0, 1.0, 0.0, 1.0],
            ],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function zeroRegionProvider() : Generator
    {
        yield [
            NumPower::array([[0.0]]),
            [[0.0]],
            [[0.0]],
        ];

        yield [
            NumPower::array([[0.5, 0.9, 0.99, 1.0, 1.01]]),
            [[0.0, 0.0, 0.0, 0.0, 1.01]],
            [[0.0, 0.0, 0.0, 0.0, 1.0]],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function extremeValuesProvider() : Generator
    {
        yield [
            NumPower::array([[10.0, 100.0, 1000.0]]),
            [[10.0, 100.0, 1000.0]],
            [[1.0, 1.0, 1.0]],
        ];

        yield [
            NumPower::array([[-10.0, -100.0, -1000.0]]),
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new ThresholdedReLU($this->threshold);
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Thresholded ReLU (threshold: 1)', (string) $this->activationFn);
    }

    #[Test]
    #[TestDox('It throws an exception when threshold is negative')]
    public function testInvalidThresholdException() : void
    {
        $this->expectException(InvalidThresholdException::class);

        new ThresholdedReLU(-1.0);
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
    #[TestDox('Correctly handles different threshold values')]
    #[DataProvider('thresholdValuesProvider')]
    public function testThresholdValues(float $threshold, NDArray $input, array $expectedActivation, array $expectedDerivative) : void
    {
        $activationFn = new ThresholdedReLU($threshold);

        $activations = $activationFn->activate($input)->toArray();
        $derivatives = $activationFn->differentiate($input)->toArray();

        static::assertEqualsWithDelta($expectedActivation, $activations, 1e-7);
        static::assertEqualsWithDelta($expectedDerivative, $derivatives, 1e-7);
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
