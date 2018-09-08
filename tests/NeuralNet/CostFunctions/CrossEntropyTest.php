<?php

namespace Rubix\Tests\NeuralNet\CostFunctions;

use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;

class CrossEntropyTest extends TestCase
{
    protected $costFunction;

    protected $expected;

    protected $activation;

    protected $delta;

    public function setUp()
    {
        $this->expected = new Matrix([[1.], [0.], [0.], [1.], [0.]]);

        $this->activation = new Matrix([[0.99], [0.2], [0.7], [0.80], [0.02]]);

        $this->delta = new Matrix([
            [0.009949832494966436],
            [0.5450311338010299],
            [1.4536452650830487],
            [0.17851484105136778],
            [0.09844316742608239],
        ]);

        $this->costFunction = new CrossEntropy();
    }

    public function test_build_cost_function()
    {
        $this->assertInstanceOf(CrossEntropy::class, $this->costFunction);
        $this->assertInstanceOf(CostFunction::class, $this->costFunction);
    }

    public function test_compute()
    {
        $cost = $this->costFunction
            ->compute($this->expected, $this->activation)
            ->asArray();

        $this->assertEquals($this->delta->asArray(), $cost);
    }

    public function test_differentiate()
    {
        $derivative = $this->costFunction
            ->differentiate($this->expected, $this->activation, $this->delta)
            ->asArray();

        $outcome = [
            [-1.01010101010101],
            [1.2499999999999998],
            [3.3333333333333326],
            [-1.25],
            [1.0204081632653061],
        ];

        $this->assertEquals($outcome, $derivative);
    }
}
