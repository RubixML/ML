<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use PHPUnit\Framework\TestCase;

#[Group('NeuralNet')]
#[CoversClass(Parameter::class)]
class ParameterTest extends TestCase
{
    protected Parameter $param;

    protected Optimizer $optimizer;

    protected function setUp() : void
    {
        $this->param = new Parameter(Matrix::quick([
            [5, 4],
            [-2, 6],
        ]));

        $this->optimizer = new Stochastic();
    }

    public function testUpdate() : void
    {
        $gradient = Matrix::quick([
            [2, 1],
            [1, -2],
        ]);

        $expected = [
            [4.98, 3.99],
            [-2.01, 6.02],
        ];

        $this->param->update(gradient: $gradient, optimizer: $this->optimizer);

        $this->assertEquals($expected, $this->param->param()->asArray());
    }
}
