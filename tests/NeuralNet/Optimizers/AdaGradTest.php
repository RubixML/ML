<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\AdaGrad;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use PHPUnit\Framework\TestCase;

class AdaGradTest extends TestCase
{
    protected $optimizer;

    public function setUp()
    {
        $this->optimizer = new AdaGrad(0.001);
    }

    public function test_build_optimizer()
    {
        $this->assertInstanceOf(AdaGrad::class, $this->optimizer);
        $this->assertInstanceOf(Adaptive::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    public function test_warm_step()
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
            [0.099, 0.599, -0.399],
            [0.501, 0.599, -0.401],
            [0.099, 0.101, -0.699],
        ];

        $this->optimizer->warm($param);

        $this->optimizer->step($param, $gradient);

        $this->assertEquals($expected, $param->w()->asArray());
    }
}
