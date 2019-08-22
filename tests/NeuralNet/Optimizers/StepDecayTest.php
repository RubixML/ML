<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Rubix\Tensor\Tensor;
use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Optimizers\StepDecay;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Rubix\ML\NeuralNet\Parameters\MatrixParam;
use PHPUnit\Framework\TestCase;
use Generator;

class StepDecayTest extends TestCase
{
    protected $optimizer;

    public function setUp()
    {
        $this->optimizer = new StepDecay(0.001);
    }

    public function test_build_optimizer()
    {
        $this->assertInstanceOf(StepDecay::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    /**
     * @dataProvider step_provider
     */
    public function test_step(Parameter $param, Tensor $gradient, array $expected)
    {
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
                [1e-5, 5e-5, -2e-5],
                [-1e-5, 2e-5, 3e-5],
                [4e-5, -1e-5, -0.0005],
            ],
        ];
    }
}
