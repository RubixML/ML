<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\RMSProp;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Optimizers
 * @covers \Rubix\ML\NeuralNet\Optimizers\RMSProp
 */
class RMSPropTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Optimizers\RMSProp
     */
    protected $optimizer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->optimizer = new RMSProp(0.001, 0.1);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(RMSProp::class, $this->optimizer);
        $this->assertInstanceOf(Adaptive::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    /**
     * @test
     * @dataProvider stepProvider
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     * @param \Tensor\Tensor<int|float> $gradient
     * @param array[] $expected
     */
    public function step(Parameter $param, Tensor $gradient, array $expected) : void
    {
        $this->optimizer->warm($param);

        $step = $this->optimizer->step($param, $gradient);

        $this->assertEquals($expected, $step->asArray());
    }

    /**
     * @return \Generator<array>
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
                [0.0031622776601683794, 0.003162277660168379, -0.0031622776601683794],
                [-0.0031622776601683794, 0.0031622776601683794, 0.0031622776601683794],
                [0.0031622776601683794, -0.0031622776601683794, -0.0031622776601683794],
            ],
        ];
    }
}
