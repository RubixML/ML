<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;
use Generator;

class ThresholdedReLUTest extends TestCase
{
    protected $activationFn;

    public function setUp()
    {
        $this->activationFn = new ThresholdedReLU(0.1);
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(ThresholdedReLU::class, $this->activationFn);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFn);
    }

    public function test_get_range()
    {
        $this->assertEquals([0., INF], $this->activationFn->range());
    }

    /**
     * @dataProvider compute_provider
     */
    public function test_compute(Matrix $input, array $expected)
    {
        $activations = $this->activationFn->compute($input)->asArray();

        $this->assertEquals($expected, $activations);
    }

    public function compute_provider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.0, 0.0, 0.0, 20.0, 0.0],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.0, 0.31, 0.0],
                [0.99, 0.0, 0.0],
                [0.0, 0.0, 0.54],
            ],
        ];
    }

    /**
     * @dataProvider differentiate_provider
     */
    public function test_differentiate(Matrix $input, Matrix $activations, array $expected)
    {
        $derivatives = $this->activationFn->differentiate($input, $activations)->asArray();

        $this->assertEquals($expected, $derivatives);
    }

    public function differentiate_provider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            Matrix::quick([
                [1.0, 0.0, 0.0, 20.0, 0.0],
            ]),
            [
                [1, 0, 0, 1, 0],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            Matrix::quick([
                [0.0, 0.31, 0.0],
                [0.99, 0.0, 0.0],
                [0.0, 0.0, 0.54],
            ]),
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
        ];
    }
}
