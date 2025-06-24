<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\GELU;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\GELU\GELU;

#[Group('ActivationFunctions')]
#[CoversClass(GELU::class)]
class GELUTest extends TestCase
{
    /**
     * @var GELU
     */
    protected GELU $activationFn;

    /**
     * @return Generator<array>
     */
    public static function computeProvider() : Generator
    {
        yield [
            NumPower::array([
                [2, 1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.9545977, 0.8411920, -0.1542859, 0.0, 20.0, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.0542690, 0.1927302, -0.1529288],
                [0.8303745, 0.0425504, -0.0146410],
                [0.0259969, -0.1568163, 0.3808940],
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
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.0829640, 0.1326301, 0.5, 1.0, -0.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.4047141, 0.7395542, 0.1388180],
                [1.0805064, 0.5636941, 0.4760706],
                [0.5398608, 0.1204533, 0.8914529],
            ],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new GELU();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('GELU', (string) $this->activationFn);
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
