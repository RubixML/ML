<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('ActivationFunctions')]
#[CoversClass(Softmax::class)]
class SoftmaxTest extends TestCase
{
    protected Softmax $activationFn;

    public static function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0], [-0.5], [0.0], [20.0], [-10.0],
            ]),
            [
                [5.6027963875928395E-9],
                [1.2501528552426345E-9],
                [2.0611536040650294E-9],
                [0.9999999910858036],
                [9.357622885424485E-14],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.19158324070294602, 0.44831331084352877, 0.18569619981694455],
                [0.5813322146601748, 0.3561999899147059, 0.29415651924235886],
                [0.22708454463687924, 0.19548669924176545, 0.5201472809406967],
            ],
        ];
    }

    public static function differentiateProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0], [-0.5], [0.0], [20.0], [-10.0],
            ]),
            Matrix::quick([
                [5.6027963875928395E-9],
                [1.2501528552426345E-9],
                [2.0611536040650294E-9],
                [0.9999999910858036],
                [9.357622885424485E-14],
            ]),
            [
                [5.602796356201512E-9],
                [1.2501528536797524E-9],
                [2.0611535998166752E-9],
                [8.91419630158617E-9],
                [9.35762288542361E-14],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            Matrix::quick([
                [0.19158324070294602, 0.44831331084352877, 0.18569619981694455],
                [0.5813322146601748, 0.3561999899147059, 0.29415651924235886],
                [0.22708454463687924, 0.19548669924176545, 0.5201472809406967],
            ]),
            [
                [0.15487910258470305, 0.24732848616404232, 0.15121312119048996],
                [0.24338507085847125, 0.22932155709946933, 0.2076284614295786],
                [0.17551715422394046, 0.157271649661325, 0.24959408707069664],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->activationFn = new Softmax();
    }

    /**
     * @param Matrix $input
     * @param list<list<float>> $expected
     */
    #[DataProvider('computeProvider')]
    public function testActivate(Matrix $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->asArray();

        $this->assertEqualsWithDelta($expected, $activations, 1e-8);
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

        $this->assertEqualsWithDelta($expected, $derivatives, 1e-8);
    }
}
