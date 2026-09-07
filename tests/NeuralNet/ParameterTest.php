<?php

namespace Rubix\ML\Tests\NeuralNet;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
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

        $this->optimizer = new Stochastic();
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
