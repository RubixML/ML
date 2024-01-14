<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group ActivationFunctions
 * @covers \Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus
 */
class SoftPlusTest extends TestCase
{
    /**
     * @var SoftPlus
     */
    protected $activationFn;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->activationFn = new SoftPlus();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(SoftPlus::class, $this->activationFn);
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
                [1.3132616875182228, 0.4740769841801067, 0.6931471805599453, 20.000000002061153, 4.5398899216870535E-5],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.6349461015956135, 0.8601118864387145, 0.47786415060626164],
                [1.3059609474567209, 0.7339469673175899, 0.6782596763414485],
                [0.7184596480132864, 0.466573094164618, 0.9991627362708937],
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
                [1.3132616875182228, 0.4740769841801067, 0.6931471805599453, 20.000000002061153, 4.5398899216870535E-5],
            ]),
            [
                [0.7880584423829144, 0.6163482688094494, 0.6666666666666666, 0.9999999979388463, 0.5000113497248023],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            Matrix::quick([
                [0.6349461015956135, 0.8601118864387145, 0.47786415060626164],
                [1.3059609474567209, 0.7339469673175899, 0.6782596763414485],
                [0.7184596480132864, 0.466573094164618, 0.9991627362708937],
            ]),
            [
                [0.6536101281900434, 0.7026840300982835, 0.6172433983573185],
                [0.7868364913554997, 0.6756708090907413, 0.6633501645777039],
                [0.6722677309178808, 0.6145723336890343, 0.7308939307469305],
            ],
        ];
    }
}
