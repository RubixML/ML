<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\GELU;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\GELU\GELU;
use PHPUnit\Framework\TestCase;
use Generator;

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
                [1.9311704635620117, 0.8411920070648193, -0.15144622325897217, 0.0, 20.0, -0.0000005960464477539062],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.0540182888507843, 0.19418436288833618, -0.15014779567718506],
                [0.8305627107620239, 0.04266344755887985, -0.014624974690377712],
                [0.02604134939610958, -0.15386739373207092, 0.3839401602745056],
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
                [1.0829640626907349, 0.1287703514099121, 0.5, 1.0, -0.00001316985708399443],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.40266361832618713, 0.7436619400978088, 0.13489001989364624],
                [1.0804662704467773, 0.5650942921638489, 0.47553694248199463],
                [0.5407461524009705, 0.11674512922763824, 0.8949908018112183],
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

        static::assertEquals($expected, $derivatives);
    }
}
