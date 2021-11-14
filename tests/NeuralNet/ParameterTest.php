<?php

namespace Rubix\ML\Tests\NeuralNet;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use PHPUnit\Framework\TestCase;

/**
 * @group NeuralNet
 * @covers \Rubix\ML\NeuralNet\Parameter
 */
class ParameterTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Parameter
     */
    protected \Rubix\ML\NeuralNet\Parameter $param;

    /**
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->param = new Parameter(Matrix::quick([
            [5, 4],
            [-2, 6],
        ]));

        $this->optimizer = new Stochastic();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Parameter::class, $this->param);
    }

    /**
     * @test
     */
    public function id() : void
    {
        $this->assertIsInt($this->param->id());
    }

    /**
     * @test
     */
    public function update() : void
    {
        $gradient = Matrix::quick([
            [2, 1],
            [1, -2],
        ]);

        $expected = [
            [4.98, 3.99],
            [-2.01, 6.02],
        ];

        $this->param->update($gradient, $this->optimizer);

        $this->assertEquals($expected, $this->param->param()->asArray());
    }
}
