<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\HyperbolicTangent;

use Generator;
use NumPower;
use NDArray;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\HyperbolicTangent\HyperbolicTangent;

#[Group('ActivationFunctions')]
#[CoversClass(HyperbolicTangent::class)]
class HyperbolicTangentTest extends TestCase
{
    /**
     * @var HyperbolicTangent
     */
    protected HyperbolicTangent $activationFn;

    /**
     * @return Generator<array>
     */
    public static function computeProvider() : Generator
    {
        yield [
            NumPower::array([
                [9.0, 2.5, 2.0, 1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [0.9999999, 0.9866142, 0.9640275, 0.7615941, -0.4621171, 0.0, 1.0, -1.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.1194273, 0.3004370, -0.4542164],
                [0.7573622, 0.0798297, -0.0299910],
                [0.0499583, -0.4776999, 0.4929879],
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
                [0.9640275, 0.7615941, -0.4621171, 0.0, 1.0, -1.0],
            ]),
            [
                [0.0706509, 0.4199743, 0.7864477, 1.0, 0.0, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.1194273, 0.3004370, -0.4542164],
                [0.7573623, 0.0797883, -0.0299912],
                [0.0499583, -0.4778087, 0.4930591],
            ]),
            [
                [0.9857371, 0.9097375, 0.7936874],
                [0.4264023, 0.9936338, 0.9991005],
                [0.9975042, 0.7716988, 0.7568927],
            ],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new HyperbolicTangent();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Hyperbolic Tangent', (string) $this->activationFn);
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
}
