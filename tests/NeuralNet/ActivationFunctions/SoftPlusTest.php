<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus;
use Rubix\ML\NeuralNet\ActivationFunctions\Rectifier;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class SoftPlusTest extends TestCase
{
    protected $input;

    protected $activationFunction;

    protected $activations;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5], [0.0], [20.0], [-10.0]]);

        $this->activations = new Matrix([
            [1.3132616875182228], [0.4740769841801067], [0.6931471805599453],
            [20.000000002061153], [4.5398899216870535E-5],
        ]);

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

        $this->assertEquals(0.7880584423829144, $derivatives[0][0]);
        $this->assertEquals(0.6163482688094494, $derivatives[1][0]);
        $this->assertEquals(0.6666666666666666, $derivatives[2][0]);
        $this->assertEquals(0.9999999979388463, $derivatives[3][0]);
        $this->assertEquals(0.5000113497248023, $derivatives[4][0]);
    }
}
