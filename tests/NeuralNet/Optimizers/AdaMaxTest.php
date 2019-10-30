<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Rubix\ML\NeuralNet\Parameters\MatrixParam;
use PHPUnit\Framework\TestCase;
use Generator;

class AdaMaxTest extends TestCase
{
    protected $optimizer;

    public function setUp()
    {
        $this->optimizer = new AdaMax(0.001, 0.1, 0.001);
    }

    public function test_build_optimizer()
    {
        $this->assertInstanceOf(AdaMax::class, $this->optimizer);
        $this->assertInstanceOf(Adaptive::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    /**
     * @dataProvider step_provider
     */
    public function test_warm_step(Parameter $param, Tensor $gradient, array $expected)
    {
        $this->optimizer->warm($param);

        $step = $this->optimizer->step($param, $gradient);

        $this->assertEquals($expected, $step->asArray());
    }

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
                [0.0010000000000000002, 0.0010000000000000005, -0.0010000000000000002],
                [-0.0010000000000000002, 0.0010000000000000002, 0.0010000000000000002],
                [0.0010000000000000002, -0.0010000000000000002, -0.0010000000000000002],
            ],
        ];
    }
}
