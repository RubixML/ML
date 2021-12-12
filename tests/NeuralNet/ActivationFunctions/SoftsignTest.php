<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\Softsign;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group ActivationFunctions
 * @covers \Rubix\ML\NeuralNet\ActivationFunctions\Softsign
 */
class SoftsignTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\Softsign
     */
    protected $activationFn;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->activationFn = new Softsign();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Softsign::class, $this->activationFn);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFn);
    }

    /**
     * @test
     * @dataProvider computeProvider
     *
     * @param \Tensor\Matrix $input
     * @param list<list<float>> $expected $expected
     */
    public function activate(Matrix $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->asArray();

        $this->assertEquals($expected, $activations);
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function computeProvider() : Generator
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

    /**
     * @test
     * @dataProvider differentiateProvider
     *
     * @param \Tensor\Matrix $input
     * @param \Tensor\Matrix $activations
     * @param list<list<float>> $expected $expected
     */
    public function differentiate(Matrix $input, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input, $activations)->asArray();

        $this->assertEquals($expected, $derivatives);
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function differentiateProvider() : Generator
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
}
