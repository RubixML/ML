<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use PHPUnit\Framework\TestCase;

class MomentumTest extends TestCase
{
    protected $optimizer;

    public function setUp()
    {
        $this->optimizer = new Momentum(0.001);
    }

    public function test_build_optimizer()
    {
        $this->assertInstanceOf(Momentum::class, $this->optimizer);
        $this->assertInstanceOf(Adaptive::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    public function test_initialize_step()
    {
        $param = new Parameter(Matrix::quick([
            [0.1, 0.6, -0.4],
            [0.5, 0.6, -0.4],
            [0.1, 0.1, -0.7],
        ]));

        $gradient = Matrix::quick([
            [0.01, 0.05, -0.02],
            [-0.01, 0.02, 0.03],
            [0.04, -0.01, -0.5],
        ]);

        $expected = [
            [0.09999000000000001, 0.59995, -0.39998],
            [0.50001, 0.59998, -0.40003],
            [0.09996000000000001, 0.10001, -0.6995],
        ];

        $this->optimizer->initialize($param);

        $this->optimizer->step($param, $gradient);

        $this->assertEquals($expected, $param->w()->asArray());
    }
}
