<?php

namespace Rubix\Tests\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus;
use Rubix\ML\NeuralNet\ActivationFunctions\Rectifier;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class SoftPlusTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5], [0.0], [20.0], [-10.0]]);

        $this->activationFunction = new SoftPlus();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(SoftPlus::class, $this->activationFunction);
        $this->assertInstanceOf(Rectifier::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_get_range()
    {
        $this->assertEquals([0.0, INF], $this->activationFunction->range());
    }

    public function test_compute()
    {
        $activations = $this->activationFunction->compute($this->input);

        $this->assertEquals(1.3132616875182228, $activations[0][0]);
        $this->assertEquals(0.4740769841801067, $activations[1][0]);
        $this->assertEquals(0.6931471805599453, $activations[2][0]);
        $this->assertEquals(20.000000002061153, $activations[3][0]);
        $this->assertEquals(4.5398899216870535E-5, $activations[4][0]);
    }

    public function test_differentiate()
    {
        $activations = $this->activationFunction->compute($this->input);

        $derivatives = $this->activationFunction->differentiate($this->input, $activations);

        $this->assertEquals(0.7880584423829144, $derivatives[0][0]);
        $this->assertEquals(0.6163482688094494, $derivatives[1][0]);
        $this->assertEquals(0.6666666666666666, $derivatives[2][0]);
        $this->assertEquals(0.9999999979388463, $derivatives[3][0]);
        $this->assertEquals(0.5000113497248023, $derivatives[4][0]);
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
