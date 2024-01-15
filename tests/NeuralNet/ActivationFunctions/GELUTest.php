<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\GELU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group ActivationFunctions
 * @covers \Rubix\ML\NeuralNet\ActivationFunctions\GELU
 */
class GELUTest extends TestCase
{
    /**
     * @var GELU
     */
    protected $activationFn;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->activationFn = new GELU();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(GELU::class, $this->activationFn);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFn);
    }

    /**
     * @test
     * @dataProvider computeProvider
     *
     * @param Matrix $input
     * @param array<array<mixed>> $expected
     */
    public function compute(Matrix $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->asArray();

        $this->assertEqualsWithDelta($expected, $activations, 1e-8);
    }

    /**
     * @return \Generator<array<mixed>>
     */
    public function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [0.841191990607477, -0.15428599017516514, 0.0, 20.0, -0.0],
            ],
        ];

        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
                [2.0, 0.5, 0.00001, -20.0, 1.0],
            ]),
            [
                [0.841191990607477, -0.15428599017516514, 0.0, 20.0, -0.0],
                [1.9545976940871754, 0.34571400982483486, 5.0000398942280396E-6, 0.0, 0.841191990607477],
            ],
        ];
    }

    /**
     * @test
     * @dataProvider differentiateProvider
     *
     * @param Matrix $input
     * @param Matrix $activations
     * @param array<array<mixed>> $expected
     */
    public function differentiate(Matrix $input, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input, $activations)->asArray();

        $this->assertEqualsWithDelta($expected, $derivatives, 1e-8);
    }

    /**
     * @return \Generator<array<mixed>>
     */
    public function differentiateProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            Matrix::quick([
                [0.7310585786300049, -0.1887703343990727, 0.0, 19.999999958776925, -0.00045397868702434395],
            ]),
            [
                [1.082963928002244, 0.13263021771495387, 0.5, 1.0, 0.0],
            ],
        ];
    }
}
