<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\HardSigmoid;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\HardSigmoid\HardSigmoid;

#[Group('ActivationFunctions')]
#[CoversClass(HardSigmoid::class)]
class HardSigmoidTest extends TestCase
{
    /**
     * @var HardSigmoid
     */
    protected HardSigmoid $activationFn;

    /**
     * @return Generator<array>
     */
    public static function computeProvider() : Generator
    {
        yield [
            NumPower::array([
                [2.5, 2.4, 2.0, 1.0, -0.5, 0.0, 20.0, -2.5, -2.4, -10.0],
            ]),
            [
                [1.0, 0.9800000190734863, 0.8999999761581421, 0.699999988079071, 0.4000000059604645, 0.5, 1.0, 0.0, 0.019999980926513672, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.47600001096725464, 0.5619999766349792, 0.4020000100135803],
                [0.6980000138282776, 0.515999972820282, 0.49399998784065247],
                [0.5099999904632568, 0.3959999978542328, 0.6079999804496765],
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
                [0.0, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.0, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [2.99, 0.08, -2.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.20000000298023224, 0.20000000298023224, 0.20000000298023224],
                [0.0, 0.20000000298023224, 0.20000000298023224],
                [0.20000000298023224, 0.20000000298023224, 0.20000000298023224],
            ],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new HardSigmoid();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('HardSigmoid', (string) $this->activationFn);
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
    #[TestDox('Correctly differentiates the output')]
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(NDArray $output, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($output)->toArray();

        static::assertEqualsWithDelta($expected, $derivatives, 1e-16);
    }
}
