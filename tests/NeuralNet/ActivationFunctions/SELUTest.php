<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\SELU;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('ActivationFunctions')]
#[CoversClass(SELU::class)]
class SELUTest extends TestCase
{
    protected SELU $activationFn;

    public static function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.0507009873554805, -0.6917581878028713, 0.0, 21.014019747109607, -1.7580195232607867],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.19880510567087464, 0.32571730608019894, -0.6810412810460496],
                [1.0401939774819255, 0.08405607898843843, -0.05195968798746372],
                [0.05253504936777403, -0.7128731573407567, 0.5673785331719595],
            ],
        ];
    }

    public static function differentiateProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            Matrix::quick([
                [1.0507009873554805, -0.6917581878028713, 0.0, 21.014019747109607, -1.7580195232607867],
            ]),
            [
                [1.0507009873554805, 1.0312683299116618, 1.7580993408473766, 1.0507009873554805, -0.08905350803294294],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            Matrix::quick([
                [-0.19880510567087464, 0.32571730608019894, -0.6810412810460496],
                [1.0401939774819255, 0.08405607898843843, -0.05195968798746372],
                [0.05253504936777403, -0.7128731573407567, 0.5673785331719595],
            ]),
            [
                [1.5492146200276782, 1.0507009873554805, 1.0425285944224512],
                [1.0507009873554805, 1.0507009873554805, 1.7035052453762658],
                [1.0507009873554805, 1.0090828105702248, 1.0507009873554805],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->activationFn = new SELU();
    }

    /**
     * @param Matrix $input
     * @param list<list<float>> $expected
     */
    #[DataProvider('computeProvider')]
    public function testActivate(Matrix $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->asArray();

        $this->assertEquals($expected, $activations);
    }

    /**
     * @param Matrix $input
     * @param Matrix $activations
     * @param list<list<float>> $expected $expected
     */
    #[DataProvider('differentiateProvider')]
    public function differentiate(Matrix $input, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate(input: $input, output: $activations)->asArray();

        $this->assertEquals($expected, $derivatives);
    }
}
