<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class ThresholdedReLUTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    protected $activations;

    public function setUp()
    {
        $this->input = Matrix::quick([[1.0], [-0.5], [0.0], [20.0], [-10.0]]);

        $this->activations = Matrix::quick([
            [1.0], [0.0], [0.0], [20.0], [0.0],
        ]);

        $this->activationFunction = new ThresholdedReLU(0.1);
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(ThresholdedReLU::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_get_range()
    {
        $this->assertEquals([0.0, INF], $this->activationFunction->range());
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

        $this->assertThat($activations[0][0], $this->logicalAnd(
            $this->greaterThanOrEqual($min), $this->lessThanOrEqual($max))
        );

        $this->assertThat($activations[1][0], $this->logicalAnd(
            $this->greaterThanOrEqual($min), $this->lessThanOrEqual($max))
        );

        $this->assertThat($activations[2][0], $this->logicalAnd(
            $this->greaterThanOrEqual($min), $this->lessThanOrEqual($max))
        );

        $this->assertThat($activations[3][0], $this->logicalAnd(
            $this->greaterThanOrEqual($min), $this->lessThanOrEqual($max))
        );

        $this->assertThat($activations[4][0], $this->logicalAnd(
            $this->greaterThanOrEqual($min), $this->lessThanOrEqual($max))
        );
    }

    public function test_differentiate()
    {
        $activations = $this->activationFunction->compute($this->input);

        $derivatives = $this->activationFunction->differentiate($this->input, $activations);

        $this->assertEquals(1.0, $derivatives[0][0]);
        $this->assertEquals(0.0, $derivatives[1][0]);
        $this->assertEquals(0.0, $derivatives[2][0]);
        $this->assertEquals(1.0, $derivatives[3][0]);
        $this->assertEquals(0.0, $derivatives[4][0]);
    }
}
