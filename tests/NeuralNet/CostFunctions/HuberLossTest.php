<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\TestCase;

class HuberLossTest extends TestCase
{
    protected $costFn;

    protected $expected;

    protected $activation;
    
    public function setUp()
    {
        $this->expected = Matrix::quick([[36.], [22.], [18.], [41.5], [38.]]);

        $this->activation = Matrix::quick([[33.98], [20.], [4.6], [44.2], [38.5]]);

        $this->costFn = new HuberLoss();
    }

    public function test_build_cost_function()
    {
        $this->assertInstanceOf(HuberLoss::class, $this->costFn);
        $this->assertInstanceOf(CostFunction::class, $this->costFn);
    }

    public function test_compute()
    {
        $cost = $this->costFn->compute($this->expected, $this->activation);

        $this->assertEquals(3.384914773928223, $cost);
    }

    public function test_differentiate()
    {
        $derivative = $this->costFn
            ->differentiate($this->expected, $this->activation)
            ->asArray();

        $expected = [
            [-0.8961947919452747],
            [-0.8944271909999159],
            [-0.9972269926097788],
            [0.9377487607237037],
            [0.4472135954999579],
        ];

        $this->assertEquals($expected, $derivative);
    }
}
