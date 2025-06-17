<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Initializers\He;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\NeuralNet\Initializers\Uniform\Uniform;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanInException;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanOutException;
use Rubix\ML\NeuralNet\Initializers\Uniform\Exceptions\InvalidBetaException;

#[Group('Initializers')]
#[CoversClass(Uniform::class)]
final class UniformTest extends TestCase
{
    /**
     * Data provider for constrictor
     *
     * @return array<string, array<string, float>>
     */
    public static function betaProvider() : array
    {
        return [
            'negative beta' => [
                'beta' => -0.1,
            ],
            'zero beta' => [
                'beta' => 0,
            ],
        ];
    }

    /**
     * Provides valid fanIn and fanOut combinations for testing matrix shape.
     *
     * @return array<string, array{fanIn: int, fanOut: int}>
     */
    public static function validShapeDimensionsProvider() : array
    {
        return [
            'fanIn and fanOut being equal' => [
                'fanIn' => 1,
                'fanOut' => 1,
            ],
            'fanIn greater than fanOut' => [
                'fanIn' => 4,
                'fanOut' => 3,
            ],
            'fanIn less than fanOut' => [
                'fanIn' => 3,
                'fanOut' => 4,
            ],
        ];
    }

    /**
     * Provides large dimensions to validate Uniform distribution.
     *
     * @return array<string, array{fanIn: int, fanOut: int}>
     */
    public static function uniformDistributionValidationProvider() : array
    {
        return [
            'small numbers' => [
                'fanIn' => 50,
                'fanOut' => 100,
                'beta' => 0.1,
            ],
            'medium numbers' => [
                'fanIn' => 100,
                'fanOut' => 200,
                'beta' => 0.2,
            ],
            'big numbers' => [
                'fanIn' => 200,
                'fanOut' => 300,
                'beta' => 0.3,
            ],
        ];
    }

    /**
     * Provides invalid fanIn and fanOut combinations to trigger exceptions.
     *
     * @return array<string, array{fanIn: int, fanOut: int}>
     */
    public static function invalidFanValuesProvider() : array
    {
        return [
            'fanIn less than 1' => [
                'fanIn' => 0,
                'fanOut' => 1,
            ],
            'fanOut less than 1' => [
                'fanIn' => 1,
                'fanOut' => 1,
            ],
            'fanIn and fanOut less than 1' => [
                'fanIn' => 0,
                'fanOut' => 0,
            ],
        ];
    }

    #[Test]
    #[TestDox('The initializer object is created correctly')]
    public function testConstructor() : void
    {
        //expect
        $this->expectNotToPerformAssertions();

        //when
        new Uniform();
    }

    #[Test]
    #[TestDox('The initializer object is throw an exception when std less than 0')]
    #[DataProvider('betaProvider')]
    public function testConstructorWithInvaditBetaThrowsAnException(float $beta) : void
    {
        //expect
        $this->expectException(InvalidBetaException::class);

        //when
        new Uniform($beta);
    }

    #[Test]
    #[TestDox('The result matrix has correct shape')]
    #[DataProvider('validShapeDimensionsProvider')]
    public function testMatrixShapeMatchesFanInAndFanOut(int $fanIn, int $fanOut) : void
    {
        //given
        $w = new Uniform()->initialize(fanIn: $fanIn, fanOut: $fanOut);

        //when
        $shape = $w->shape();

        //then
        $this->assertSame([$fanOut, $fanIn], $shape);
    }

    #[Test]
    #[TestDox('The resulting values matches Uniform distribution')]
    #[DataProvider('uniformDistributionValidationProvider')]
    public function testDistributionStatisticsMatchUniform(int $fanIn, int $fanOut, float $beta) : void
    {
        //when
        $w = new Uniform($beta)->initialize(fanIn: $fanIn, fanOut:  $fanOut);
        $values = array_merge(...$w->toArray());

        //then
        $this->assertGreaterThanOrEqual(-$beta, min($values));
        $this->assertLessThanOrEqual($beta, max($values));
    }

    #[Test]
    #[TestDox('An exception is thrown during initialization')]
    #[DataProvider('invalidFanValuesProvider')]
    public function testExceptionThrownForInvalidFanValues(int $fanIn, int $fanOut) : void
    {
        //expect
        if ($fanIn < 1) {
            $this->expectException(InvalidFanInException::class);
        } elseif ($fanOut < 1) {
            $this->expectException(InvalidFanOutException::class);
        } else {
            $this->expectNotToPerformAssertions();
        }

        //when
        new Uniform()->initialize(fanIn: $fanIn, fanOut: $fanOut);
    }

    #[Test]
    #[TestDox('It returns correct string representation')]
    public function testToStringReturnsCorrectValue() : void
    {
        //when
        $string = (string) new Uniform();

        //then
        $this->assertEquals('Uniform (beta: 0.5)', $string);
    }
}
