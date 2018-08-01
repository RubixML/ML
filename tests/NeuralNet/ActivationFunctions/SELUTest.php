<?php

namespace Rubix\Tests\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\SELU;
use Rubix\ML\NeuralNet\ActivationFunctions\Rectifier;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class SELUTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5], [0.0], [20.0], [-10.0]]);

        $this->activationFunction = new SELU(1.05070, 1.67326);
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(SELU::class, $this->activationFunction);
        $this->assertInstanceOf(Rectifier::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_get_range()
    {
        $this->assertEquals([-1.7580942819999998, INF], $this->activationFunction->range());
    }

    public function test_compute()
    {
        $activations = $this->activationFunction->compute($this->input);

        $this->assertEquals(1.0507, $activations[0][0]);
        $this->assertEquals(-0.6917561973015314, $activations[1][0]);
        $this->assertEquals(0.0, $activations[2][0]);
        $this->assertEquals(21.014, $activations[3][0]);
        $this->assertEquals(-1.7580144646430813, $activations[4][0]);
    }

    public function test_differentiate()
    {
        $activations = $this->activationFunction->compute($this->input);

        $derivatives = $this->activationFunction->differentiate($this->input, $activations);

        $this->assertEquals(1.0507, $derivatives[0][0]);
        $this->assertEquals(0.27317176349528105, $derivatives[1][0]);
        $this->assertEquals(1.0, $derivatives[2][0]);
        $this->assertEquals(1.05070, $derivatives[3][0]);
        $this->assertEquals(-0.8471457980004855, $derivatives[4][0]);
    }

    public function test_within_range()
    {
        list($min, $max) = $this->activationFunction->range();

        $activations = $this->activationFunction->compute($this->input);

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
}
