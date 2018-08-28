<?php

namespace Rubix\Tests\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Optimizers\AdaGrad;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use PHPUnit\Framework\TestCase;

class AdaGradTest extends TestCase
{
    protected $optimizer;

    protected $gradients;

    protected $parameter;

    public function setUp()
    {
        $this->parameter = new Parameter(new Matrix([
            [0.1, 0.6, -0.4],
            [0.5, 0.6, -0.4],
            [0.1, 0.1, -0.7],
        ]));

        $this->gradients = new Matrix([
            [0.01, 0.05, -0.02],
            [-0.01, 0.02, 0.03],
            [0.04, -0.01, -0.5],
        ]);

        $this->optimizer = new AdaGrad(0.001);
    }

    public function test_build_optimizer()
    {
        $this->assertInstanceOf(AdaGrad::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    public function test_step()
    {
        $step = $this->optimizer->step($this->parameter, $this->gradients);

        $this->assertInstanceOf(Matrix::class, $step);
    }
}
