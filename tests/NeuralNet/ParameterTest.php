<?php

namespace Rubix\ML\Tests\NeuralNet;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('NeuralNet')]
#[CoversClass(Parameter::class)]
class ParameterTest extends TestCase
{
    /**
     * @var Parameter
     */
    protected Parameter $param;

    /**
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer;

    protected function setUp() : void
    {
        $this->param = new Parameter(Matrix::quick([
            [5, 4],
            [-2, 6],
        ]));

        $this->optimizer = new Stochastic(new Constant());
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Parameter::class, $this->param);
    }

    #[Test]
    public function id() : void
    {
        $this->assertIsInt($this->param->id());
    }

    #[Test]
    public function accumulateGradient() : void
    {
        $gradient = Matrix::quick([
            [2, 1],
            [1, -2],
        ]);

        $this->assertNull($this->param->gradient());

        $this->param->accumulate($gradient);

        $this->assertInstanceOf(Matrix::class, $this->param->gradient());
        $this->assertEquals($gradient->asArray(), $this->param->gradient()->asArray());

        $this->param->accumulate($gradient);

        $expected = [
            [4, 2],
            [2, -4],
        ];

        $this->assertEquals($expected, $this->param->gradient()->asArray());
    }

    #[Test]
    public function resetGradient() : void
    {
        $this->param->accumulate(Matrix::quick([
            [2, 1],
            [1, -2],
        ]));

        $this->param->resetGradient();

        $this->assertNull($this->param->gradient());
    }

    #[Test]
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

        $this->param->accumulate($gradient);

        $this->param->update($this->optimizer);

        $this->assertEquals($expected, $this->param->param()->asArray());
    }
}
