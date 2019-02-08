<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\Softsign;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class SoftsignTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    protected $activations;

    public function setUp()
    {
        $this->input = Matrix::quick([[1.0], [-0.5], [0.0], [20.0], [-10.0]]);

        $this->activations = Matrix::quick([
            [0.5], [-0.3333333333333333], [0.0], [0.9523809523809523],
            [-0.9090909090909091],
        ]);

        $this->activationFunction = new Softsign();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(Softsign::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_get_range()
    {
        $this->assertEquals([-1.0, 1.0], $this->activationFunction->range());
    }

    public function test_compute()
    {
        [$min, $max] = $this->activationFunction->range();

        $activations = $this->activationFunction->compute($this->input);

        $this->assertEquals($this->activations[0][0], $activations[0][0]);
        $this->assertEquals($this->activations[1][0], $activations[1][0]);
        $this->assertEquals($this->activations[2][0], $activations[2][0]);
        $this->assertEquals($this->activations[3][0], $activations[3][0]);
        $this->assertEquals($this->activations[4][0], $activations[4][0]);

        $this->assertThat(
            $activations[0][0],
            $this->logicalAnd(
            $this->greaterThanOrEqual($min),
            $this->lessThanOrEqual($max)
        )
        );

        $this->assertThat(
            $activations[1][0],
            $this->logicalAnd(
            $this->greaterThanOrEqual($min),
            $this->lessThanOrEqual($max)
        )
        );

        $this->assertThat(
            $activations[2][0],
            $this->logicalAnd(
            $this->greaterThanOrEqual($min),
            $this->lessThanOrEqual($max)
        )
        );

        $this->assertThat(
            $activations[3][0],
            $this->logicalAnd(
            $this->greaterThanOrEqual($min),
            $this->lessThanOrEqual($max)
        )
        );

        $this->assertThat(
            $activations[4][0],
            $this->logicalAnd(
            $this->greaterThanOrEqual($min),
            $this->lessThanOrEqual($max)
        )
        );
    }

    public function test_differentiate()
    {
        $derivatives = $this->activationFunction->differentiate($this->input, $this->activations);

        $this->assertEquals(0.25, $derivatives[0][0]);
        $this->assertEquals(0.4444444444444444, $derivatives[1][0]);
        $this->assertEquals(1.0, $derivatives[2][0]);
        $this->assertEquals(0.0022675736961451248, $derivatives[3][0]);
        $this->assertEquals(0.008264462809917356, $derivatives[4][0]);
    }
}
