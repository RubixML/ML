<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\HardSiLU;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\HardSiLU\HardSiLU;

#[Group('ActivationFunctions')]
#[CoversClass(HardSiLU::class)]
class HardSiLUTest extends TestCase
{
    /**
     * @var HardSiLU
     */
    protected HardSiLU $activationFn;

    /**
     * @return Generator<array>
     */
    public static function computeProvider() : Generator
    {
        yield [
            NumPower::array([
                [2.5, 2.0, 1.0, -0.5, 0.0, 20.0, -2.5, -10.0],
            ]),
            [
                [2.5, 1.7999999, 0.6999999, -0.2000000, 0.0, 20.0, 0.0, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.0571199, 0.1742199, -0.1969800],
                [0.6910200, 0.0412799, -0.0148199],
                [0.0254999, -0.2059199, 0.3283199],
            ],
        ];

        // Boundary test cases
        yield [
            NumPower::array([
                // Exact boundary values for HardSigmoid (x = -2.5, x = 2.5)
                [-2.5, 2.5],
                // Values just inside boundaries
                [-2.499, 2.499],
                // Values just outside boundaries
                [-2.501, 2.501],
            ]),
            [
                // At x = -2.5, HardSigmoid(x) = 0, so HardSiLU(-2.5) = -2.5 * 0 = 0
                // At x = 2.5, HardSigmoid(x) = 1, so HardSiLU(2.5) = 2.5 * 1 = 2.5
                [0.0, 2.5],
                // Just inside boundaries
                [-0.0004997, 2.4985003],
                // Just outside boundaries
                [0.0, 2.5009999],
            ],
        ];

        // Zero and near-zero test cases
        yield [
            NumPower::array([
                // Zero and very small values around zero
                [0.0, 0.000001, -0.0000001, 0.0000000001, -0.0000000001],
            ]),
            [
                // HardSiLU(0) = 0 * 0.5 = 0
                // For very small values, HardSigmoid(x) ≈ 0.5, so HardSiLU(x) ≈ x * 0.5
                [0.0, 0.0000005, -0.0000000, 0.0000000, -0.0000000],
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
                [2.5, 1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.5, 0.8999999, 0.30000001192092896, 0.5, 1.0, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.4520000, 0.6239999, 0.3040000],
                [0.8960000, 0.5319999, 0.4879999],
                [0.5199999, 0.2919999, 0.7159999],
            ],
        ];

        // Boundary test cases for differentiation
        yield [
            NumPower::array([
                // Exact boundary values for HardSigmoid (x = -2.5, x = 2.5)
                [-2.5, 2.5],
                // Values just inside boundaries
                [-2.499, 2.499],
                // Values just outside boundaries
                [-2.501, 2.501],
            ]),
            [
                // At boundaries: derivative is 0 at x = -2.5 and 1 at x = 2.5
                [-0.5, 1.5],
                // Just inside boundaries
                [-0.4996000, 1.4996000],
                // Just outside boundaries
                [0.0, 1.0],
            ],
        ];

        // Zero and near-zero test cases for differentiation
        yield [
            NumPower::array([
                // Zero and very small values around zero
                [0.0, -0.00001, 0.000001, -0.0000001, 0.00000001, -0.000000001],
            ]),
            [
                // At x = 0, derivative is 0.5
                // For very small values, derivative is close to 0.5
                [0.5, 0.4999960, 0.5000003, 0.4999999, 0.5, 0.5],
            ],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new HardSiLU();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('HardSiLU', (string) $this->activationFn);
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
}
