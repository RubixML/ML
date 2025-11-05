<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Parameters;

use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Stochastic\Stochastic;
use PHPUnit\Framework\TestCase;

#[Group('Parameters')]
#[CoversClass(Parameter::class)]
class ParameterTest extends TestCase
{
    protected Parameter $param;

    protected Optimizer $optimizer;

    protected function setUp() : void
    {
        $this->param = new Parameter(NumPower::array([
            [5, 4],
            [-2, 6],
        ]));

        $this->optimizer = new Stochastic();
    }

    public function testUpdate() : void
    {
        $gradient = NumPower::array([
            [2, 1],
            [1, -2],
        ]);

        $expected = [
            [4.98, 3.99],
            [-2.01, 6.02],
        ];

        $this->param->update(gradient: $gradient, optimizer: $this->optimizer);

        self::assertEqualsWithDelta($expected, $this->param->param()->toArray(), 1e-7);
    }
}
