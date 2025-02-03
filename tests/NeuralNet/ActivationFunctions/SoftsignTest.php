<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\Softsign;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('ActivationFunctions')]
#[CoversClass(Softsign::class)]
class SoftsignTest extends TestCase
{
    protected Softsign $activationFn;

    public static function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [0.5, -0.3333333333333333, 0.0, 0.9523809523809523, -0.9090909090909091],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.10714285714285712, 0.23664122137404578, -0.32885906040268453],
                [0.49748743718592964, 0.07407407407407407, -0.029126213592233007],
                [0.047619047619047616, -0.34210526315789475, 0.35064935064935066],
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
                [0.5, -0.3333333333333333, 0.0, 0.9523809523809523, -0.9090909090909091],
            ]),
            [
                [0.25, 0.4444444444444444, 1.0, 0.0022675736961451248, 0.008264462809917356],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            Matrix::quick([
                [-0.10714285714285712, 0.23664122137404578, -0.32885906040268453],
                [0.49748743718592964, 0.07407407407407407, -0.029126213592233007],
                [0.047619047619047616, -0.34210526315789475, 0.35064935064935066],
            ]),
            [
                [0.7971938775510203, 0.5827166249053085, 0.4504301608035674],
                [0.252518875785965, 0.8573388203017832, 0.9425959091337544],
                [0.9070294784580498, 0.4328254847645429, 0.42165626581210996],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->activationFn = new Softsign();
    }

    /**
     * @param Matrix $input
     * @param list<list<float>> $expected $expected
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
     * @param list<list<float>> $expected
     */
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(Matrix $input, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate(input: $input, output: $activations)->asArray();

        $this->assertEquals($expected, $derivatives);
    }
}
