<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group ActivationFunctions
 * @covers \Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU
 */
class LeakyReLUTest extends TestCase
{
    /**
     * @var LeakyReLU
     */
    protected $activationFn;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->activationFn = new LeakyReLU(0.01);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(LeakyReLU::class, $this->activationFn);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFn);
    }

    /**
     * @test
     * @dataProvider computeProvider
     *
     * @param Matrix $input
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
                [1.0, -0.005, 0.0, 20.0, -0.1],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.0012, 0.31, -0.0049],
                [0.99, 0.08, -0.0003],
                [0.05, -0.005200000000000001, 0.54],
            ],
        ];
    }

    /**
     * @test
     * @dataProvider differentiateProvider
     *
     * @param Matrix $input
     * @param Matrix $activations
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
                [1.0, -0.005, 0.0, 20.0, -0.1],
            ]),
            [
                [1.0, 0.01, 0.01, 1.0, 0.01],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            Matrix::quick([
                [-0.0012, 0.31, -0.0049],
                [0.99, 0.08, -0.0003],
                [0.05, -0.005200000000000001, 0.54],
            ]),
            [
                [0.01, 1.0, 0.01],
                [1.0, 1.0, 0.01],
                [1.0, 0.01, 1.0],
            ],
        ];
    }
}
