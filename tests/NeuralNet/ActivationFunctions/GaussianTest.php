<?php

namespace Rubix\Tests\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\Gaussian;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;

class GaussianTest extends TestCase
{
    const TOLERANCE = 1e-10;

    protected $input;

    protected $activationFunction;

    protected $activations;

    public function setUp()
    {
        $this->input = new Matrix([[1.0], [-0.5], [0.0], [20.0], [-10.0]]);

        $this->activations = new Matrix([
            [0.36787944117144233], [0.7788007830714049], [1.0],
            [1.9151695967140057E-174], [3.720075976020836E-44],
        ]);

        $this->activationFunction = new Gaussian();
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(Gaussian::class, $this->activationFunction);
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

        $this->assertEquals(-0.7357588823428847, $derivatives[0][0], '', self::TOLERANCE);
        $this->assertEquals(0.7788007830714049, $derivatives[1][0], '', self::TOLERANCE);
        $this->assertEquals(-0.0, $derivatives[2][0], '', self::TOLERANCE);
        $this->assertEquals(-7.660678386856023E-173, $derivatives[3][0], '', self::TOLERANCE);
        $this->assertEquals(7.440151952041672E-43, $derivatives[4][0], '', self::TOLERANCE);
    }
}
