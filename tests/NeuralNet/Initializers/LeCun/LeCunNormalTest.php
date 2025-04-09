<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Initializers\He;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\NeuralNet\Initializers\LeCun\LeCunNormal;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanInException;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanOutException;

#[Group('Initializers')]
#[CoversClass(LeCunNormal::class)]
final class LeCunNormalTest extends TestCase
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
     * Provides large dimensions to validate mean and standard deviation for Le Cun normal distribution.
     *
     * @return array<string, array{fanIn: int, fanOut: int}>
     */
    public static function leCunNormalDistributionValidationProvider() : array
    {
        return [
            'small numbers' => [
                'fanIn' => 30,
                'fanOut' => 10,
            ],
            'medium numbers' => [
                'fanIn' => 300,
                'fanOut' => 100,
            ],
            'big numbers' => [
                'fanIn' => 3000,
                'fanOut' => 1000,
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
        new LeCunNormal();
    }

    #[Test]
    #[TestDox('The result matrix has correct shape')]
    #[DataProvider('validShapeDimensionsProvider')]
    public function testMatrixShapeMatchesFanInAndFanOut(int $fanIn, int $fanOut) : void
    {
        //given
        $w = new LeCunNormal()->initialize(fanIn: $fanIn, fanOut: $fanOut);

        //when
        $shape = $w->shape();

        //then
        $this->assertSame([$fanOut, $fanIn], $shape);
    }

    #[Test]
    #[TestDox('The resulting values matches distribution Le Cun (normal distribution)')]
    #[DataProvider('leCunNormalDistributionValidationProvider')]
    public function testDistributionStatisticsMatchLeCunNormal(int $fanIn, int $fanOut) : void
    {
        //given
        $expectedStd = sqrt(1 / $fanOut);
        $w = new LeCunNormal()->initialize(fanIn: $fanIn, fanOut:  $fanOut);
        $flatValues = array_merge(...$w->toArray());

        //when
        $mean = array_sum($flatValues) / count($flatValues);
        $variance = array_sum(array_map(fn ($x) => ($x - $mean) ** 2, $flatValues)) / count($flatValues);
        $std = sqrt($variance);

        //then
        $this->assertThat(
            $mean,
            $this->logicalAnd(
                $this->greaterThan(-0.1),
                $this->lessThan(0.1)
            ),
            'Mean is not within the expected range'
        );
        $this->assertThat(
            $std,
            $this->logicalAnd(
                $this->greaterThan($expectedStd * 0.9),
                $this->lessThan($expectedStd * 1.1)
            ),
            'Standard deviation does not match Le Cun initialization'
        );
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
        new LeCunNormal()->initialize(fanIn: $fanIn, fanOut: $fanOut);
    }

    #[Test]
    #[TestDox('String representation is correct')]
    public function testToStringReturnsCorrectValue() : void
    {
        //when
        $string = (string) new LeCunNormal();

        //then
        $this->assertEquals('Le Cun Normal', $string);
    }
}
