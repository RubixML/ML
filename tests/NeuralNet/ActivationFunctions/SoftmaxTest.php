<?php

namespace Rubix\Tests\NeuralNet\ActivationFunctions;

use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class SoftmaxTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    protected $activations;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5], [0.0], [20.0], [-10.0]]);

        $this->activations = new Matrix([
            [5.6027963875928395E-9], [1.2501528552426345E-9], [2.0611536040650294E-9],
            [0.9999999910858036], [9.357622885424485E-14],
        ]);

        $this->activationFunction = new Softmax();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(Softmax::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_get_range()
    {
        $this->assertEquals([0.0, 1.0], $this->activationFunction->range());
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

        $this->assertEquals(5.602796356201512E-9, $derivatives[0][0]);
        $this->assertEquals(1.2501528552426345E-9, $derivatives[1][0]);
        $this->assertEquals(2.0611536040650294E-9, $derivatives[2][0]);
        $this->assertEquals(8.91419630158617E-9, $derivatives[3][0]);
        $this->assertEquals(9.357622885424485E-14, $derivatives[4][0]);
    }
}
