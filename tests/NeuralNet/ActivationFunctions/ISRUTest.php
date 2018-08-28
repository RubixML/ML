<?php

namespace Rubix\Tests\NeuralNet\ActivationFunctions;

use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ISRU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class ISRUTest extends TestCase
{
    const TOLERANCE = 1e-10;

    protected $input;

    protected $activationFunction;

    protected $activations;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5], [0.0], [20.0], [-10.0]]);

        $this->activations = new Matrix([
            [0.7071067811865475], [-0.4472135954999579], [0.0],
            [0.9987523388778445], [-0.9950371902099892],
        ]);

        $this->activationFunction = new ISRU(1.0);
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(ISRU::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_get_range()
    {
        $this->assertEquals([-1.0, 1.0], $this->activationFunction->range());
    }

    public function test_compute()
    {
        list($min, $max) = $this->activationFunction->range();

        $activations = $this->activationFunction->compute($this->input);

        $this->assertEquals($this->activations[0][0], $activations[0][0], '', self::TOLERANCE);
        $this->assertEquals($this->activations[1][0], $activations[1][0], '', self::TOLERANCE);
        $this->assertEquals($this->activations[2][0], $activations[2][0], '', self::TOLERANCE);
        $this->assertEquals($this->activations[3][0], $activations[3][0], '', self::TOLERANCE);
        $this->assertEquals($this->activations[4][0], $activations[4][0], '', self::TOLERANCE);

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

        $this->assertEquals(0.3535533905932737, $derivatives[0][0], '', self::TOLERANCE);
        $this->assertEquals(0.7155417527999326, $derivatives[1][0], '', self::TOLERANCE);
        $this->assertEquals(1.0, $derivatives[2][0], '', self::TOLERANCE);
        $this->assertEquals(0.0001245327105832724, $derivatives[3][0], '', self::TOLERANCE);
        $this->assertEquals(0.0009851853368415735, $derivatives[4][0], '', self::TOLERANCE);
    }
}
