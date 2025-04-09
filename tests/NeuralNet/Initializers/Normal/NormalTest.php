<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Initializers\He;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\NeuralNet\Initializers\Normal\Normal;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanInException;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanOutException;
use Rubix\ML\NeuralNet\Initializers\Normal\Exceptions\InvalidStandardDeviationException;

#[Group('Initializers')]
#[CoversClass(Normal::class)]
final class NormalTest extends TestCase
{
    /**
     * Data provider for testConstructorThrowsForInvalidStd
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
            'fanIn equals fanOut' => [
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
    public static function normalDistributionInitializationProvider() : array
    {
        return [
            'small matrix' => [
                'fanIn' => 80,
                'fanOut' => 50,
                'stdDev' => 0.25
            ],
            'medium matrix' => [
                'fanIn' => 300,
                'fanOut' => 100,
                'stdDev' => 0.5,
            ],
            'large matrix' => [
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
            'fanIn is zero' => [
                'fanIn' => 0,
                'fanOut' => 1,
            ],
            'fanOut is zero' => [
                'fanIn' => 1,
                'fanOut' => 0,
            ],
            'both fanIn and fanOut are zero' => [
                'fanIn' => 0,
                'fanOut' => 0,
            ],
        ];
    }

    #[Test]
    #[TestDox('It constructs the initializer with default standard deviation')]
    public function testConstructorSucceedsWithDefaultStdDev() : void
    {
        //expect
        $this->expectNotToPerformAssertions();

        //when
        new Normal();
    }

    #[Test]
    #[TestDox('It throws an exception if standard deviation is not positive')]
    #[DataProvider('invalidStandardDeviationProvider')]
    public function testConstructorThrowsForInvalidStdDev(float $stdDev) : void
    {
        //expect
        $this->expectException(InvalidStandardDeviationException::class);

        //when
        new Normal($stdDev);
    }

    #[Test]
    #[TestDox('The initialized matrix has the correct shape')]
    #[DataProvider('validFanInFanOutCombinationsProvider')]
    public function testInitializedMatrixHasCorrectShape(int $fanIn, int $fanOut) : void
    {
        //given
        $w = new Normal()->initialize(fanIn: $fanIn, fanOut: $fanOut);

        //when
        $shape = $w->shape();

        //then
        $this->assertSame([$fanOut, $fanIn], $shape);
    }

    #[Test]
    #[TestDox('The initialized values follow a Normal distribution')]
    #[DataProvider('normalDistributionInitializationProvider')]
    public function testValuesFollowNormalDistribution(int $fanIn, int $fanOut, float $stdDev) : void
    {
        //given
        $w = new Normal($stdDev)->initialize(fanIn: $fanIn, fanOut:  $fanOut);
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
            'Standard deviation does not match Normal initialization'
        );
    }

    #[Test]
    #[TestDox('It throws an exception if fanIn or fanOut are less than 1')]
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
        new Normal()->initialize(fanIn: $fanIn, fanOut: $fanOut);
    }

    #[Test]
    #[TestDox('String representation is correct')]
    public function testToStringReturnsExpectedFormat() : void
    {
        //when
        $string = (string) new Normal();

        //then
        $this->assertEquals('Normal (stdDev: 0.05)', $string);
    }
}
