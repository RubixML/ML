<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\RMSProp;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use PHPUnit\Framework\TestCase;

class RMSPropTest extends TestCase
{
    protected $optimizer;

    public function setUp()
    {
        $this->optimizer = new RMSProp(0.001);
    }

    public function test_build_optimizer()
    {
        $this->assertInstanceOf(RMSProp::class, $this->optimizer);
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
            [0.09683772233983162, 0.5968377223398316, -0.39683772233983167],
            [0.5031622776601684, 0.5968377223398316, -0.4031622776601684],
            [0.09683772233983162, 0.10316227766016839, -0.6968377223398315],
        ];

        $this->optimizer->initialize($param);
        
        $this->optimizer->step($param, $gradient);

        $this->assertEquals($expected, $param->w()->asArray());
    }
}
