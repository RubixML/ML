<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;
use Generator;

class ReLUTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\ReLU
     */
    protected $activationFn;

    public function setUp() : void
    {
        $this->activationFn = new ReLU();
    }

    public function test_build_activation_function() : void
    {
        $this->assertInstanceOf(ReLU::class, $this->activationFn);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFn);
    }

    /**
     * @param \Tensor\Matrix $input
     * @param array[] $expected
     *
     * @dataProvider compute_provider
     */
    public function test_compute(Matrix $input, array $expected) : void
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
                [0.99, 0.08, 0.0],
                [0.05, 0.0, 0.54],
            ],
        ];
    }

    /**
     * @param \Tensor\Matrix $input
     * @param \Tensor\Matrix $activations
     * @param array[] $expected
     *
     * @dataProvider differentiate_provider
     */
    public function test_differentiate(Matrix $input, Matrix $activations, array $expected) : void
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
                [0.99, 0.08, 0.0],
                [0.05, 0.0, 0.54],
            ]),
            [
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 1],
            ],
        ];
    }
}
