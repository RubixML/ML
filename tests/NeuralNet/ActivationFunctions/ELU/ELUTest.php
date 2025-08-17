<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\ELU;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU\ELU;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU\Exceptions\InvalidAlphaException;

#[Group('ActivationFunctions')]
#[CoversClass(ELU::class)]
class ELUTest extends TestCase
{
    /**
     * @var ELU
     */
    protected ELU $activationFn;

    /**
     * @return Generator<array>
     */
    public static function computeProvider() : Generator
    {
        yield [
            NumPower::array([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.0, -0.3934693, 0.0, 20.0, -0.9999545],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.1130795, 0.3100000, -0.3873736],
                [0.9900000, 0.0799999, -0.0295544],
                [0.0500000, -0.4054794, 0.5400000],
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
                [1.0, 0.6065306, 1.0, 1.0, 0.0000454],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.8869204, 1.0, 0.6126263],
                [1.0, 1.0, 0.9704455],
                [1.0, 0.5945205, 1.0],
            ],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new ELU(1.0);
    }

    #[Test]
    #[TestDox('Can be constructed with valid alpha parameter')]
    public function testConstructorWithValidAlpha() : void
    {
        $activationFn = new ELU(2.0);

        static::assertInstanceOf(ELU::class, $activationFn);
        static::assertEquals('ELU (alpha: 2)', (string) $activationFn);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with invalid alpha parameter')]
    public function testConstructorWithInvalidAlpha() : void
    {
        $this->expectException(InvalidAlphaException::class);

        new ELU(-346);
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('ELU (alpha: 1)', (string) $this->activationFn);
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
    #[TestDox('Correctly differentiates the input using buffered output')]
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(NDArray $input, array $expected) : void
    {
        $output = $this->activationFn->activate($input);
        $derivatives = $this->activationFn->differentiate($input, $output)->toArray();

        static::assertEqualsWithDelta($expected, $derivatives, 1e-7);
    }
}
