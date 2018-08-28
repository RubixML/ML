<?php

namespace Rubix\Tests\NeuralNet\ActivationFunctions;

use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\Rectifier;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class LeakyReLUTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    protected $activations;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5], [0.0], [20.0], [-10.0]]);

        $this->activations = new Matrix([
            [1.0], [-0.005], [0.0], [20.0], [-0.1],
        ]);

        $this->activationFunction = new LeakyReLU(0.01);
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(LeakyReLU::class, $this->activationFunction);
        $this->assertInstanceOf(Rectifier::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_get_range()
    {
        $this->assertEquals([-INF, INF], $this->activationFunction->range());
    }

    public function test_compute()
    {
        list($min, $max) = $this->activationFunction->range();

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
        $derivatives = $this->activationFunction->differentiate($this->input, $this->activations);

        $this->assertEquals(1.0, $derivatives[0][0]);
        $this->assertEquals(0.01, $derivatives[1][0]);
        $this->assertEquals(0.01, $derivatives[2][0]);
        $this->assertEquals(1.0, $derivatives[3][0]);
        $this->assertEquals(0.01, $derivatives[4][0]);
    }
}
