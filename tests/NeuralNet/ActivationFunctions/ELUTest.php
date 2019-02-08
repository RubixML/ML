<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class ELUTest extends TestCase
{
    const TOLERANCE = 1e-10;

    protected $input;

    protected $activationFunction;

    protected $activations;

    public function setUp()
    {
        $this->input = Matrix::quick([[1.0], [-0.5], [0.0], [20.0], [-10.0]]);

        $this->activations = Matrix::quick([
            [1.0], [-0.3934693402873666], [0.0], [20.0], [-0.9999546000702375],
        ]);

        $this->activationFunction = new ELU(1.0);
    }

    public function test_build_activation_function()
    {
        $this->assertInstanceOf(ELU::class, $this->activationFunction);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFunction);
    }

    public function test_bad_alpha_parameter()
    {
        $this->expectException(InvalidArgumentException::class);

        new ELU(-346);
    }

    public function test_get_range()
    {
        $this->assertEquals([-1.0, INF], $this->activationFunction->range());
    }

    public function test_compute()
    {
        [$min, $max] = $this->activationFunction->range();

        $activations = $this->activationFunction->compute($this->input);

        $this->assertEquals($this->activations[0][0], $activations[0][0], '', self::TOLERANCE);
        $this->assertEquals($this->activations[1][0], $activations[1][0], '', self::TOLERANCE);
        $this->assertEquals($this->activations[2][0], $activations[2][0], '', self::TOLERANCE);
        $this->assertEquals($this->activations[3][0], $activations[3][0], '', self::TOLERANCE);
        $this->assertEquals($this->activations[4][0], $activations[4][0], '', self::TOLERANCE);

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

        $this->assertEquals(1.0, $derivatives[0][0], '', self::TOLERANCE);
        $this->assertEquals(0.6065306597126334, $derivatives[1][0], '', self::TOLERANCE);
        $this->assertEquals(1.0, $derivatives[2][0], '', self::TOLERANCE);
        $this->assertEquals(1.0, $derivatives[3][0], '', self::TOLERANCE);
        $this->assertEquals(4.539992976249074E-5, $derivatives[4][0], '', self::TOLERANCE);
    }
}
