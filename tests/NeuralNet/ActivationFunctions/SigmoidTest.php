<?php

namespace Rubix\Tests\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class SigmoidTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5], [0.0], [20.0], [-10.0]]);

        $this->activationFunction = new Sigmoid();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(Sigmoid::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_get_range()
    {
        $this->assertEquals([0.0, 1.0], $this->activationFunction->range());
    }

    public function test_compute()
    {
        $activations = $this->activationFunction->compute($this->input);

        $this->assertEquals(0.7310585786300049, $activations[0][0]);
        $this->assertEquals(0.3775406687981454, $activations[1][0]);
        $this->assertEquals(0.5, $activations[2][0]);
        $this->assertEquals(0.9999999979388463, $activations[3][0]);
        $this->assertEquals(4.5397868702434395E-5, $activations[4][0]);
    }

    public function test_differentiate()
    {
        $activations = $this->activationFunction->compute($this->input);

        $derivatives = $this->activationFunction->differentiate($this->input, $activations);

        $this->assertEquals(0.19661193324148185, $derivatives[0][0]);
        $this->assertEquals(0.2350037122015945, $derivatives[1][0]);
        $this->assertEquals(0.25, $derivatives[2][0]);
        $this->assertEquals(2.0611536879193953E-9, $derivatives[3][0]);
        $this->assertEquals(4.5395807735951673E-5, $derivatives[4][0]);
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
