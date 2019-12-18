<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Rubix\ML\NeuralNet\Parameters\MatrixParam;
use PHPUnit\Framework\TestCase;
use Generator;

class MomentumTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Optimizers\Momentum
     */
    protected $optimizer;

    public function setUp() : void
    {
        $this->optimizer = new Momentum(0.001, 0.1);
    }

    public function test_build_optimizer() : void
    {
        $this->assertInstanceOf(Momentum::class, $this->optimizer);
        $this->assertInstanceOf(Adaptive::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    /**
     * @param \Rubix\ML\NeuralNet\Parameters\Parameter $param
     * @param \Tensor\Tensor<int|float> $gradient
     * @param array[] $expected
     *
     * @dataProvider step_provider
     */
    public function test_warm_step(Parameter $param, Tensor $gradient, array $expected) : void
    {
        $this->optimizer->warm($param);

        $step = $this->optimizer->step($param, $gradient);

        $this->assertEquals($expected, $step->asArray());
    }

    /**
     * @return \Generator<array>
     */
    public function step_provider() : Generator
    {
        yield [
            new MatrixParam(Matrix::quick([
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
                [1e-5, 5e-5, -2e-5],
                [-1e-5, 2e-5, 3e-5],
                [4e-5, -1e-5, -0.0005],
            ],
        ];
    }
}
