<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\GELU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
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
     * @return Generator<array<mixed>>
     */
    public static function computeProvider() : Generator
    {
        yield [
            Matrix::fromArray([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ], false),
            [
                [0.841344746068543, -0.15426876936299344, 0.0, 20.0, -0.0],
            ],
        ];

        yield [
            Matrix::fromArray([
                [1.0, -0.5, 0.0, 20.0, -10.0],
                [2.0, 0.5, 0.00001, -20.0, 1.0],
            ], false),
            [
                [0.841344746068543, -0.15426876936299344, 0.0, 20.0, -0.0],
                [1.9544997361036416, 0.34573123063700656, 5.00003989422804E-6, -0.0, 0.841344746068543],
            ],
        ];
    }

    /**
     * @return Generator<array<mixed>>
     */
    public static function differentiateProvider() : Generator
    {
        yield [
            Matrix::fromArray([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ], false),
            Matrix::fromArray([
                [0.841344746068543, -0.15426876936299344, 0.0, 20.0, -0.0],
            ], false),
            [
                [1.0833154705876864, 0.13250487534383712, 0.5, 1.0, -7.694598626706419E-22],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->activationFn = new GELU();
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(GELU::class, $this->activationFn);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFn);
    }

    /**
     * @param Matrix $x
     * @param array<array<mixed>> $expected
     */
    #[DataProvider('computeProvider')]
    #[Test]
    public function compute(Matrix $x, array $expected) : void
    {
        $activations = $this->activationFn->activate($x)->asArray();

        $this->assertEqualsWithDelta($expected, $activations, 1e-8);
    }

    /**
     * @param Matrix $x
     * @param Matrix $activations
     * @param array<array<mixed>> $expected
     */
    #[DataProvider('differentiateProvider')]
    #[Test]
    public function differentiate(Matrix $x, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($x, $activations)->asArray();

        $this->assertEqualsWithDelta($expected, $derivatives, 1e-8);
    }
}
