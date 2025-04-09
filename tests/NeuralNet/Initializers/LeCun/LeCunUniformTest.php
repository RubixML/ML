<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Initializers\He;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\NeuralNet\Initializers\LeCun\LeCunUniform;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanInException;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanOutException;

#[Group('Initializers')]
#[CoversClass(LeCunUniform::class)]
final class LeCunUniformTest extends TestCase
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
            ]
        ];
    }

    /**
     * Provides large dimensions to validate Le Cun uniform distribution.
     *
     * @return array<string, array{fanIn: int, fanOut: int}>
     */
    public static function leCunUniformDistributionValidationProvider() : array
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
            ]
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
        new LeCunUniform();
    }

    #[Test]
    #[TestDox('The result matrix has correct shape')]
    #[DataProvider('validShapeDimensionsProvider')]
    public function testMatrixShapeMatchesFanInAndFanOut(int $fanIn, int $fanOut) : void
    {
        //given
        $w = new LeCunUniform()->initialize(fanIn: $fanIn, fanOut: $fanOut);

        //when
        $shape = $w->shape();

        //then
        $this->assertSame([$fanOut, $fanIn], $shape);
    }

    #[Test]
    #[TestDox('The resulting values matches distribution Le Cun (uniform distribution)')]
    #[DataProvider('leCunUniformDistributionValidationProvider')]
    public function testDistributionStatisticsMatchLeCunUniform(int $fanIn, int $fanOut) : void
    {
        //given
        $limit = sqrt(3 / $fanOut);

        //when
        $w = new LeCunUniform()->initialize(fanIn: $fanIn, fanOut:  $fanOut);
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
        new LeCunUniform()->initialize(fanIn: $fanIn, fanOut: $fanOut);
    }

    #[Test]
    #[TestDox('It returns correct string representation')]
    public function testToStringReturnsCorrectValue() : void
    {
        //when
        $string = (string) new LeCunUniform();

        //then
        $this->assertEquals('Le Cun Uniform', $string);
    }
}
