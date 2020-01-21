<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group ActivationFunctions
 * @covers \Rubix\ML\NeuralNet\ActivationFunctions\SiLU
 */
class SiLUTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\SiLU
     */
    protected $activationFn;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->activationFn = new SiLU();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(SiLU::class, $this->activationFn);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFn);
    }

    /**
     * @test
     * @dataProvider computeProvider
     *
     * @param \Tensor\Matrix $input
     * @param array[] $expected
     */
    public function compute(Matrix $input, array $expected) : void
    {
        $activations = $this->activationFn->compute($input)->asArray();

        $this->assertEquals($expected, $activations);
    }

    /**
     * @return \Generator<array>
     */
    public function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [0.7310585786300049, -0.1887703343990727, 0.0, 19.999999958776925, -0.00045397868702434395],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.056404313788251385, 0.17883443095093435, -0.18614784815188584],
                [ 0.7217970431258135, 0.04159914721244655, -0.014775016873481388],
                [0.025624869824210517, -0.1938831615171383, 0.3411787055774949],
            ],
        ];
    }

    /**
     * @test
     * @dataProvider differentiateProvider
     *
     * @param \Tensor\Matrix $input
     * @param \Tensor\Matrix $activations
     * @param array[] $expected
     */
    public function differentiate(Matrix $input, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input, $activations)->asArray();

        $this->assertEquals($expected, $derivatives);
    }

    /**
     * @return \Generator<array>
     */
    public function differentiateProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            Matrix::quick([
                [1.0507009873554805, -0.6917581878028713, 0.0, 21.014019747109607, -1.7580195232607867],
            ]),
            [
                [1.013635595604245, -0.05305067013503684, 0.5, 1.000000041251969, -1.757894315052591],
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
                [0.3646763889226077, 0.714701054038943, -0.04242451141092363],
                [1.0108890339486736, 0.5603371540947567, 0.46613105002043614],
                [0.5381083698268307, -0.07422457460408638, 0.8407141480931224],
            ],
        ];
    }
}
