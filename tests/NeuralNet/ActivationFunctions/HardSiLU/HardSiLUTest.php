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
                [2.5, 1.7999999523162842, 0.699999988079071, -0.20000000298023224, 0.0, 20.0, 0.0, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.05711999908089638, 0.1742199957370758, -0.19698001444339752],
                [0.6910200119018555, 0.04127999767661095, -0.014819999225437641],
                [0.025499999523162842, -0.2059199959039688, 0.3283199965953827],
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
                [-0.0004997340147383511, 2.498500347137451],
                // Just outside boundaries
                [0.0, 2.500999927520752],
            ],
        ];

        // Zero and near-zero test cases
        yield [
            NumPower::array([
                // Zero and very small values around zero
                [0.0, 0.0000001, -0.0000001, 0.0000000001, -0.0000000001],
            ]),
            [
                // HardSiLU(0) = 0 * 0.5 = 0
                // For very small values, HardSigmoid(x) ≈ 0.5, so HardSiLU(x) ≈ x * 0.5
                [0.0, 0.00000005000000058430487, -0.00000004999999703159119, 0.0000000000500000006675716, -0.0000000000500000006675716],
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
                [1.0, 0.8999999761581421, 0.30000001192092896, 0.5, 1.0, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.45200002193450928, 0.6239999532699585, 0.30400002002716064],
                [0.8960000276565552, 0.531999945640564, 0.48799997568130493],
                [0.5199999809265137, 0.2919999957084656, 0.715999960899353],
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
                [0.0, 1.0],
                // Just inside boundaries
                [-0.49960005283355713, 1.4996000528335571],
                // Just outside boundaries
                [0.0, 1.0],
            ],
        ];

        // Zero and near-zero test cases for differentiation
        yield [
            NumPower::array([
                // Zero and very small values around zero
                [0.0, 0.0000001, -0.0000001, 0.0000000001, -0.0000000001],
            ]),
            [
                // At x = 0, derivative is 0.5
                // For very small values, derivative is close to 0.5
                [0.5, 0.5, 0.4999999403953552, 0.5, 0.5],
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
}
