<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus;

#[Group('ActivationFunctions')]
#[CoversClass(SoftPlus::class)]
class SoftPlusTest extends TestCase
{
    /**
     * @var SoftPlus
     */
    protected SoftPlus $activationFn;

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
                [2.1269280910491943, 1.31326162815094, 0.4740769863128662, 0.6931471824645996, 20.0000000, 4.541770613286644E-5],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.6349461078643799, 0.8601119518280029, 0.4778640866279602],
                [1.305961012840271, 0.7339470386505127, 0.6782596707344055],
                [0.7184596061706543, 0.4665731191635132, 0.9991626739501953],
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
                [0.8807970285415649, 0.7310585975646973, 0.3775406777858734, 0.5000000, 1.0000000, 4.539787187241018E-5],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.47003597021102905, 0.5768852829933167, 0.37989357113838196],
                [0.7290879487991333, 0.5199893712997437, 0.49250054359436035],
                [0.5124973654747009, 0.37285223603248596, 0.6318124532699585],
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
            [[0.6931471824645996]],
            [[0.5000000]],
        ];

        // Test very small positive values
        yield [
            NumPower::array([[1e-15, 1e-10, 1e-7]]),
            [[0.6931471824645996, 0.6931471824645996, 0.6931471824645996]],
            [[0.5000000, 0.5000000, 0.5000000596046448]],
        ];

        // Test very small negative values
        yield [
            NumPower::array([[-1e-15, -1e-10, -1e-7]]),
            [[0.6931471824645996, 0.6931471824645996, 0.6931471228599548]],
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
            [[10.000045776367188, 20.0000000, 50.0000000]],
            [[0.9999545812606812, 1.0000000, 1.0000000]],
        ];

        // Test with large negative values
        yield [
            NumPower::array([[-10.0, -20.0, -50.0]]),
            [[4.541770613286644E-5, 0.0000000, 0.0000000]],
            [[0.0000454, 0.0000000, 0.0000000]],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new SoftPlus();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('SoftPlus', (string) $this->activationFn);
    }

    #[Test]
    #[TestDox('Correctly activates the input')]
    #[DataProvider('computeProvider')]
    public function testActivate(NDArray $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();

        $this->assertEqualsWithDelta($expected, $activations, 1e-8);
    }

    #[Test]
    #[TestDox('Correctly differentiates the input')]
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(NDArray $input, array $expected) : void
    {
        $output = $this->activationFn->activate($input);
        $derivatives = $this->activationFn->differentiate($input, $output)->toArray();

        static::assertEqualsWithDelta($expected, $derivatives, 1e-8);
    }

    #[Test]
    #[TestDox('Correctly handles values around zero')]
    #[DataProvider('zeroRegionProvider')]
    public function testZeroRegion(NDArray $input, array $expectedActivation, array $expectedDerivative) : void
    {
        $output = $this->activationFn->activate($input);
        $activations = $output->toArray();
        $derivatives = $this->activationFn->differentiate($input, $output)->toArray();

        static::assertEqualsWithDelta($expectedActivation, $activations, 1e-8);
        static::assertEqualsWithDelta($expectedDerivative, $derivatives, 1e-8);
    }

    #[Test]
    #[TestDox('Correctly handles extreme values')]
    #[DataProvider('extremeValuesProvider')]
    public function testExtremeValues(NDArray $input, array $expectedActivation, array $expectedDerivative) : void
    {
        $output = $this->activationFn->activate($input);
        $activations = $output->toArray();
        $derivatives = $this->activationFn->differentiate($input, $output)->toArray();

        static::assertEqualsWithDelta($expectedActivation, $activations, 1e-8);
        static::assertEqualsWithDelta($expectedDerivative, $derivatives, 1e-8);
    }
}
