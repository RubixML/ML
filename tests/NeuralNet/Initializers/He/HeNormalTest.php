<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Initializers\He;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\NeuralNet\Initializers\He\HeNormal;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanInException;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanOutException;

#[Group('Initializers')]
#[CoversClass(HeNormal::class)]
final class HeNormalTest extends TestCase
{
    /**
     * Provides valid fanIn and fanOut combinations for testing matrix shape.
     *
     * @return array<string, array{fanIn: int, fanOut: int}>
     */
    public static function validShapeDimensionsProvider() : array
    {
        return [
            'equal fanIn and fanOut' => ['fanIn' => 1, 'fanOut' => 1],
            'fanIn greater than fanOut' => ['fanIn' => 4, 'fanOut' => 3],
            'fanIn less than fanOut' => ['fanIn' => 3, 'fanOut' => 4],
        ];
    }

    /**
     * Provides large dimensions to validate mean and standard deviation for He normal distribution.
     *
     * @return array<string, array{fanIn: int, fanOut: int}>
     */
    public static function heNormalDistributionValidationProvider() : array
    {
        return [
            'small numbers' => ['fanIn' => 30, 'fanOut' => 10],
            'medium numbers' => ['fanIn' => 300, 'fanOut' => 100],
            'large numbers' => ['fanIn' => 3000, 'fanOut' => 1000],
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
            'fanIn less than 1' => ['fanIn' => 0, 'fanOut' => 1],
            'fanOut less than 1' => ['fanIn' => 1, 'fanOut' => 0],
            'both fanIn and fanOut invalid' => ['fanIn' => 0, 'fanOut' => 0],
        ];
    }

    #[Test]
    #[TestDox('It constructs the HeNormal initializer without errors')]
    public function testConstructor() : void
    {
        //expect
        $this->expectNotToPerformAssertions();

        //when
        new HeNormal();
    }

    #[Test]
    #[TestDox('It creates a matrix of correct shape based on fanIn and fanOut')]
    #[DataProvider('validShapeDimensionsProvider')]
    public function testMatrixShapeMatchesFanInAndFanOut(int $fanIn, int $fanOut) : void
    {
        //given
        $matrix = new HeNormal()->initialize(fanIn: $fanIn, fanOut: $fanOut);

        //when
        $shape = $matrix->shape();

        //then
        $this->assertSame([$fanOut, $fanIn], $shape);
    }

    #[Test]
    #[TestDox('It generates values with mean ~0 and std ~sqrt(2 / fanOut)')]
    #[DataProvider('heNormalDistributionValidationProvider')]
    public function testDistributionStatisticsMatchHeNormal(int $fanIn, int $fanOut) : void
    {
        //given
        $expectedStd = sqrt(2 / $fanOut);
        $matrix = new HeNormal()->initialize(fanIn: $fanIn, fanOut: $fanOut);
        $flatValues = array_merge(...$matrix->toArray());

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
            'Mean is not within expected range'
        );
        $this->assertThat(
            $std,
            $this->logicalAnd(
                $this->greaterThan($expectedStd * 0.9),
                $this->lessThan($expectedStd * 1.1)
            ),
            'Standard deviation is not within acceptable He initialization range'
        );
    }

    #[Test]
    #[TestDox('It throws an exception when fanIn or fanOut is less than 1')]
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
        new HeNormal()->initialize(fanIn: $fanIn, fanOut: $fanOut);
    }

    #[Test]
    #[TestDox('It returns correct string representation')]
    public function testToStringReturnsCorrectValue() : void
    {
        //when
        $string = (string) new HeNormal();

        //then
        $this->assertEquals('He Normal', $string);
    }
}
