<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\ELU;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU\ELU;
use PHPUnit\Framework\TestCase;
use Generator;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU\Exceptions\InvalidAlphaException;

/**
 * @group ActivationFunctions
 * @covers \Rubix\ML\NeuralNet\ActivationFunctions\ELU\ELU
 */
#[Group('ActivationFunctions')]
#[CoversClass(ELU::class)]
class ELUTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\ELU\ELU
     */
    protected ELU $activationFn;

    /**
     * @return \Generator<array<mixed>>
     */
    public static function computeProvider() : Generator
    {
        yield [
            NumPower::array([
                [1.0, -0.5, 0.0, 20.0, -10.0]
            ]),
            [
                [1.0, -0.39346933364868164, 0.0, 20.0, -0.9999545812606812],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.11307956278324127, 0.3100000023841858, -0.3873736262321472],
                [0.9900000095367432, 0.07999999821186066, -0.029554465785622597],
                [0.05000000074505806, -0.40547943115234375, 0.5400000214576721],
            ],
        ];
    }

    /**
     * @return \Generator<array<mixed>>
     */
    public static function differentiateProvider() : Generator
    {
        yield [
            NumPower::array([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.0, 0.6065306663513184, 1.0, 1.0, 4.539993096841499E-5],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.8869204521179199, 1.0, 0.6126263737678528],
                [1.0, 1.0, 0.9704455137252808],
                [1.0, 0.5945205688476562, 1.0],
            ],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        $this->activationFn = new ELU(1.0);
    }

    /**
     * @test
     */
    public function testConstructorWithValidAlpha() : void
    {
        $activationFn = new ELU(2.0);

        $this->assertInstanceOf(ELU::class, $activationFn);
        $this->assertEquals('ELU (alpha: 2)', (string) $activationFn);
    }

    /**
     * @test
     */
    public function testConstructorWithInvalidAlpha() : void
    {
        $this->expectException(InvalidAlphaException::class);

        new ELU(-346);
    }

    /**
     * @test
     */
    public function testToString() : void
    {
        $this->assertEquals('ELU (alpha: 1)', (string) $this->activationFn);
    }

    /**
     * @param NDArray $input
     * @param list<list<float>> $expected
     */
    #[DataProvider('computeProvider')]
    public function testActivate(NDArray $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();

        $this->assertEquals($expected, $activations);
    }

    /**
     * @param NDArray $input
     * @param list<list<float>> $expected
     */
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate1(NDArray $input, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input)->toArray();

        $this->assertEquals($expected, $derivatives);
    }
}
