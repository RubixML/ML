<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;

class HuberLossTest extends TestCase
{
    protected $costFunction;

    protected $expected;

    protected $activation;

    protected $delta;

    public function setUp()
    {
        $this->expected = new Matrix([[36.], [22.], [18.], [41.5], [38.]]);

        $this->activation = new Matrix([[33.98], [20.], [4.6], [44.2], [38.5]]);

        $this->delta = new Matrix([
            [1.2539742678211772],
            [1.2360679774997898],
            [12.43726162579266],
            [1.8792360097775962],
            [0.1180339887498949],
        ]);

        $this->costFunction = new HuberLoss();
    }

    public function test_build_cost_function()
    {
        $this->assertInstanceOf(HuberLoss::class, $this->costFunction);
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
            [-0.8961947919452747],
            [-0.8944271909999159],
            [-0.9972269926097788],
            [0.9377487607237037],
            [0.4472135954999579],
        ];

        $this->assertEquals($outcome, $derivative);
    }
}
