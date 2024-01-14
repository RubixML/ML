<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\AdaGrad;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Optimizers
 * @covers \Rubix\ML\NeuralNet\Optimizers\
 */
class AdaGradTest extends TestCase
{
    /**
     * @var AdaGrad
     */
    protected $optimizer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->optimizer = new AdaGrad(0.001);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(AdaGrad::class, $this->optimizer);
        $this->assertInstanceOf(Adaptive::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    /**
     * @test
     * @dataProvider stepProvider
     *
     * @param Parameter $param
     * @param \Tensor\Tensor<int|float> $gradient
     * @param list<list<float>> $expected
     */
    public function step(Parameter $param, Tensor $gradient, array $expected) : void
    {
        $this->optimizer->warm($param);

        $step = $this->optimizer->step($param, $gradient);

        $this->assertEquals($expected, $step->asArray());
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function stepProvider() : Generator
    {
        yield [
            new Parameter(Matrix::quick([
                [0.1, 0.6, -0.4],
                [0.5, 0.6, -0.4],
                [0.1, 0.1, -0.7],
            ])),
            Matrix::quick([
                [0.01, 0.05, -0.02],
                [-0.01, 0.02, 0.03],
                [0.04, -0.01, -0.5],
            ]),
            [
                [0.001, 0.001, -0.001],
                [-0.001, 0.001, 0.001],
                [0.001, -0.001, -0.001],
            ],
        ];
    }
}
