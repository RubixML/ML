<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\NeuralNet\Initializers\Xavier2Uniform;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\InvalidFanInException;
use Rubix\ML\Exceptions\InvalidFanOutException;

#[Group('Initializers')]
#[CoversClass(Xavier2Uniform::class)]
final class Xavier2UniformTest extends TestCase
{
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
     * Provides large dimensions to validate Xavier uniform distribution.
     *
     * @return array<string, array{fanIn: int, fanOut: int}>
     */
    public static function xavier2UniformDistributionValidationProvider() : array
    {
        return [
            'small numbers' => [
                'fanIn' => 50,
                'fanOut' => 100,
            ],
            'medium numbers' => [
                'fanIn' => 100,
                'fanOut' => 200,
            ],
            'big numbers' => [
                'fanIn' => 200,
                'fanOut' => 300,
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
                'fanOut' => 0,
            ],
            'fanIn and fanOut less than 1' => [
                'fanIn' => 0,
                'fanOut' => 0,
            ],
        ];
    }

    #[Test]
    #[TestDox('The initializer object is created correctly')]
    public function constructor() : void
    {
        //expect
        $this->expectNotToPerformAssertions();

        //when
        new Xavier2Uniform();
    }

    #[Test]
    #[TestDox('The result matrix has correct shape')]
    #[DataProvider('validShapeDimensionsProvider')]
    public function matrixShapeMatchesFanInAndFanOut(int $fanIn, int $fanOut) : void
    {
        //given
        $w = (new Xavier2Uniform())->initialize(fanIn: $fanIn, fanOut: $fanOut, dataType: 'float32');

        //when
        $shape = $w->shape();

        //then
        $this->assertSame([$fanOut, $fanIn], $shape);
    }

    #[Test]
    #[TestDox('The resulting values matches distribution Xavier (uniform distribution)')]
    #[DataProvider('xavier2UniformDistributionValidationProvider')]
    public function distributionStatisticsMatchXavier2Uniform(int $fanIn, int $fanOut) : void
    {
        //given
        $limit = (6.0 / ($fanOut + $fanIn)) ** 0.25;

        //when
        $w = (new Xavier2Uniform())->initialize(fanIn: $fanIn, fanOut: $fanOut, dataType: 'float32');
        $values = array_merge(...$w->toArray());

        //then
        $bins = array_fill(0, 10, 0);

        foreach ($values as $value) {
            $normalizedValue = ($value + $limit) / (2 * $limit);
            $bin = (int) ($normalizedValue * 10);

            if ($bin >= 10) {
                $bin = 9;
            }

            ++$bins[$bin];
        }

        $expectedCount = count($values) / 10;
        $tolerance = 0.15 * $expectedCount;

        $this->assertGreaterThanOrEqual(-$limit, min($values));
        $this->assertLessThanOrEqual($limit, max($values));

        foreach ($bins as $count) {
            $this->assertGreaterThanOrEqual($expectedCount - $tolerance, $count);
            $this->assertLessThanOrEqual($expectedCount + $tolerance, $count);
        }
    }

    #[Test]
    #[TestDox('An exception is thrown during initialization')]
    #[DataProvider('invalidFanValuesProvider')]
    public function exceptionThrownForInvalidFanValues(int $fanIn, int $fanOut) : void
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
        (new Xavier2Uniform())->initialize(fanIn: $fanIn, fanOut: $fanOut, dataType: 'float32');
    }

    #[Test]
    #[TestDox('It returns correct string representation')]
    public function toStringReturnsCorrectValue() : void
    {
        //when
        $string = (string) new Xavier2Uniform();

        //then
        $this->assertEquals('Xavier-2 Uniform', $string);
    }
}
