<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use Generator;

class ELUTest extends TestCase
{
    protected $activationFn;

    public function setUp()
    {
        $this->activationFn = new ELU(1.0);
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(ELU::class, $this->activationFn);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFn);
    }

    public function test_bad_alpha_parameter()
    {
        $this->expectException(InvalidArgumentException::class);

        new ELU(-346);
    }

    public function test_get_range()
    {
        $this->assertEquals([-1.0, INF], $this->activationFn->range());
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
                [1.0, -0.5, 0.0, 20.0, -10.0]
            ]),
            [
                [1.0, -0.3934693402873666, 0.0, 20.0, -0.9999546000702375],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.11307956328284252, 0.31, -0.3873736058155839],
                [0.99, 0.08, -0.029554466451491845],
                [0.05, -0.4054794520298056, 0.54],
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
                [1.0, -0.3934693402873666, 0.0, 20.0, -0.9999546000702375],
            ]),
            [
                [1.0, 0.6065306597126334, 1.0, 1.0, 4.539992976249074E-5],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            Matrix::quick([
                [-0.11307956328284252, 0.31, -0.3873736058155839],
                [0.99, 0.08, -0.029554466451491845],
                [0.05, -0.4054794520298056, 0.54],
            ]),
            [
                [0.8869204367171575, 1.0, 0.6126263941844161],
                [1.0, 1.0, 0.9704455335485082],
                [1.0, 0.5945205479701944, 1.0],
            ],
        ];
    }
}
