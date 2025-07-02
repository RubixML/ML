<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\SiLU;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU\SiLU;

#[Group('ActivationFunctions')]
#[CoversClass(SiLU::class)]
class SiLUTest extends TestCase
{
    /**
     * @var SiLU
     */
    protected SiLU $activationFn;

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
                    1.7615940, // 2.0 * sigmoid(2.0)
                    0.7310586, // 1.0 * sigmoid(1.0)
                    -0.1887703, // -0.5 * sigmoid(-0.5)
                    0.0, // 0.0 * sigmoid(0.0)
                    19.9999980, // 20.0 * sigmoid(20.0)
                    -0.0000454, // -10.0 * sigmoid(-10.0)
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
                    -0.0564043, // -0.12 * sigmoid(-0.12)
                    0.1788344, // 0.31 * sigmoid(0.31)
                    -0.1861478, // -0.49 * sigmoid(-0.49)
                ],
                [
                    0.7217970, // 0.99 * sigmoid(0.99)
                    0.0415991, // 0.08 * sigmoid(0.08)
                    -0.0147750, // -0.03 * sigmoid(-0.03)
                ],
                [
                    0.0256249, // 0.05 * sigmoid(0.05)
                    -0.1938832, // -0.52 * sigmoid(-0.52)
                    0.3411787, // 0.54 * sigmoid(0.54)
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
                [2.0, 1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            NumPower::array([
                [1.7615940, 0.7310586, -0.1887703, 0.0, 19.9999980, -0.0000454],
            ]),
            [
                [
                    1.0839772, // sigmoid(2.0) + 2.0 * sigmoid(2.0) * (1 - sigmoid(2.0))
                    0.9277255, // sigmoid(1.0) + 1.0 * sigmoid(1.0) * (1 - sigmoid(1.0))
                    0.2350038, // sigmoid(-0.5) - 0.5 * sigmoid(-0.5) * (1 - sigmoid(-0.5))
                    0.5000000, // sigmoid(0.0) + 0.0 * sigmoid(0.0) * (1 - sigmoid(0.0))
                    1.0000000, // sigmoid(20.0) + 20.0 * sigmoid(20.0) * (1 - sigmoid(20.0))
                    -0.0004536, // sigmoid(-10.0) - 10.0 * sigmoid(-10.0) * (1 - sigmoid(-10.0))
                ],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            NumPower::array([
                [-0.0564043, 0.1788344, -0.1861478],
                [0.7217970, 0.0415991, -0.0147750],
                [0.0256249, -0.1938832, 0.3411787],
            ]),
            [
                [
                    0.4401437, // sigmoid(-0.12) - 0.12 * sigmoid(-0.12) * (1 - sigmoid(-0.12))
                    0.6525527, // sigmoid(0.31) + 0.31 * sigmoid(0.31) * (1 - sigmoid(0.31))
                    0.2644620, // sigmoid(-0.49) - 0.49 * sigmoid(-0.49) * (1 - sigmoid(-0.49))
                ],
                [
                    0.9246314, // sigmoid(0.99) + 0.99 * sigmoid(0.99) * (1 - sigmoid(0.99))
                    0.5399574, // sigmoid(0.08) + 0.08 * sigmoid(0.08) * (1 - sigmoid(0.08))
                    0.4850022, // sigmoid(-0.03) - 0.03 * sigmoid(-0.03) * (1 - sigmoid(-0.03))
                ],
                [
                    0.5249895, // sigmoid(0.05) + 0.05 * sigmoid(0.05) * (1 - sigmoid(0.05))
                    0.2512588, // sigmoid(-0.52) - 0.52 * sigmoid(-0.52) * (1 - sigmoid(-0.52))
                    0.7574301, // sigmoid(0.54) + 0.54 * sigmoid(0.54) * (1 - sigmoid(0.54))
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
            [[0.0]], // 0.0 * sigmoid(0.0) = 0.0 * 0.5 = 0.0
            [[0.5]], // sigmoid(0.0) + 0.0 * sigmoid'(0.0) = 0.5 + 0.0 * 0.25 = 0.5
        ];

        // Test very small positive values
        yield [
            NumPower::array([[1e-15, 1e-10, 1e-7]]),
            [[5e-16, 5e-11, 5e-8]], // x * sigmoid(x) ≈ x * 0.5 for small x
            [[0.5, 0.5, 0.5]], // sigmoid(x) + x * sigmoid'(x) ≈ 0.5 + x * 0.25 ≈ 0.5 for small x
        ];

        // Test very small negative values
        yield [
            NumPower::array([[-1e-15, -1e-10, -1e-7]]),
            [[-5e-16, -5e-11, -5e-8]], // x * sigmoid(x) ≈ x * 0.5 for small x
            [[0.5, 0.5, 0.5]], // sigmoid(x) + x * sigmoid'(x) ≈ 0.5 + x * 0.25 ≈ 0.5 for small x
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
            [[9.9995460, 20.0, 50.0]], // x * sigmoid(x) ≈ x for large positive x
            [[1.0004087, 1.0, 1.0]], // sigmoid(x) + x * sigmoid'(x) ≈ 1.0 for large positive x
        ];

        // Test with large negative values
        yield [
            NumPower::array([[-10.0, -20.0, -50.0]]),
            [[-0.0004539, -0.0, -0.0]], // x * sigmoid(x) ≈ 0 for large negative x
            [[-0.0004085, -0.0, -0.0]], // sigmoid(x) + x * sigmoid'(x) ≈ 0 for large negative x
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new SiLU();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('SiLU', (string) $this->activationFn);
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
    #[TestDox('Correctly differentiates the input and output')]
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(NDArray $input, NDArray $output, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input, $output)->toArray();

        static::assertEqualsWithDelta($expected, $derivatives, 1e-7);
    }

    #[Test]
    #[TestDox('Correctly handles values around zero')]
    #[DataProvider('zeroRegionProvider')]
    public function testZeroRegion(NDArray $input, array $expectedActivation, array $expectedDerivative) : void
    {
        $output = $this->activationFn->activate($input);
        $activations = $output->toArray();
        $derivatives = $this->activationFn->differentiate($input, $output)->toArray();

        static::assertEqualsWithDelta($expectedActivation, $activations, 1e-7);
        static::assertEqualsWithDelta($expectedDerivative, $derivatives, 1e-7);
    }

    #[Test]
    #[TestDox('Correctly handles extreme values')]
    #[DataProvider('extremeValuesProvider')]
    public function testExtremeValues(NDArray $input, array $expectedActivation, array $expectedDerivative) : void
    {
        $output = $this->activationFn->activate($input);
        $activations = $output->toArray();
        $derivatives = $this->activationFn->differentiate($input, $output)->toArray();

        static::assertEqualsWithDelta($expectedActivation, $activations, 1e-7);
        static::assertEqualsWithDelta($expectedDerivative, $derivatives, 1e-7);
    }
}
