<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Initializers\He;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\NeuralNet\Initializers\Normal\TruncatedNormal;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanInException;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanOutException;
use Rubix\ML\NeuralNet\Initializers\Normal\Exceptions\InvalidStandardDeviationException;

#[Group('Initializers')]
#[CoversClass(TruncatedNormal::class)]
final class TruncatedNormalTest extends TestCase
{
    /**
     * Data provider for testConstructorThrowsForInvalidStdDev
     *
     * @return array<string, array<string, float>>
     */
    public static function invalidStandardDeviationProvider() : array
    {
        return [
            'negative stdDev' => [
                'stdDev' => -0.1,
            ],
            'zero stdDev' => [
                'stdDev' => 0,
            ]
        ];
    }

    /**
     * Data provider for testInitializedMatrixHasCorrectShape
     *
     * @return array<string, array<string, int>>
     */
    public static function validFanInFanOutCombinationsProvider() : array
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
     * Data provider for testValuesFollowNormalDistribution
     *
     * @return array<string, array<string, float|int>>
     */
    public static function truncatedNormalDistributionInitializationProvider() : array
    {
        return [
            'small numbers' => [
                'fanIn' => 30,
                'fanOut' => 10,
                'stdDev' => 0.25
            ],
            'medium numbers' => [
                'fanIn' => 300,
                'fanOut' => 100,
                'stdDev' => 0.5,
            ],
            'big numbers' => [
                'fanIn' => 3000,
                'fanOut' => 1000,
                'stdDev' => 1.75
            ]
        ];
    }

    /**
     * Data provider for testInitializationThrowsForInvalidFanValues
     *
     * @return array<string, array<string, int>>
     */
    public static function invalidFanInFanOutProvider() : array
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
    public function testConstructorSucceedsWithDefaultStdDev() : void
    {
        //expect
        $this->expectNotToPerformAssertions();

        //when
        new TruncatedNormal();
    }

    #[Test]
    #[TestDox('The initializer object is throw an exception when stdDev less than 0')]
    #[DataProvider('invalidStandardDeviationProvider')]
    public function testConstructorThrowsForInvalidStdDev(float $stdDev) : void
    {
        //expect
        $this->expectException(InvalidStandardDeviationException::class);

        //when
        new TruncatedNormal($stdDev);
    }

    #[Test]
    #[TestDox('The result matrix has correct shape')]
    #[DataProvider('validFanInFanOutCombinationsProvider')]
    public function testInitializedMatrixHasCorrectShape(int $fanIn, int $fanOut) : void
    {
        //given
        $w = new TruncatedNormal()->initialize(fanIn: $fanIn, fanOut: $fanOut);

        //when
        $shape = $w->shape();

        //then
        $this->assertSame([$fanOut, $fanIn], $shape);
    }

    #[Test]
    #[TestDox('The resulting values matches distribution Truncated Normal')]
    #[DataProvider('truncatedNormalDistributionInitializationProvider')]
    public function testValuesFollowTruncatedNormalDistribution(int $fanIn, int $fanOut, float $stdDev) : void
    {
        //given
        $w = new TruncatedNormal($stdDev)->initialize(fanIn: $fanIn, fanOut:  $fanOut);
        $flatValues = array_merge(...$w->toArray());

        //when
        $mean = array_sum($flatValues) / count($flatValues);
        $variance = array_sum(array_map(fn ($x) => ($x - $mean) ** 2, $flatValues)) / count($flatValues);
        $resultStd = sqrt($variance);

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
            $resultStd,
            $this->logicalAnd(
                $this->greaterThan($stdDev * 0.9),
                $this->lessThan($stdDev * 1.1)
            ),
            'Standard deviation does not match Truncated Normal initialization'
        );
        $this->assertLessThanOrEqual(
            $stdDev * 2.3,
            max($flatValues),
            'Maximum value does not match Truncated Normal initialization'
        );
        $this->assertGreaterThanOrEqual(
            $stdDev * -2.3,
            min($flatValues),
            'Minimum value does not match Truncated Normal initialization'
        );
    }

    #[Test]
    #[TestDox('An exception is thrown during initialization')]
    #[DataProvider('invalidFanInFanOutProvider')]
    public function testInitializationThrowsForInvalidFanValues(int $fanIn, int $fanOut) : void
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
        new TruncatedNormal()->initialize(fanIn: $fanIn, fanOut: $fanOut);
    }

    #[Test]
    #[TestDox('String representation is correct')]
    public function testToStringReturnsExpectedFormat() : void
    {
        //when
        $string = (string) new TruncatedNormal();

        //then
        $this->assertEquals('Truncated Normal (stdDev: 0.05)', $string);
    }
}
