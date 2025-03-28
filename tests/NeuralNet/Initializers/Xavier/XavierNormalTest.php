<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Initializers\He;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\NeuralNet\Initializers\Xavier\XavierNormal;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanInException;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanOutException;

#[Group('Initializers')]
#[CoversClass(XavierNormal::class)]
final class XavierNormalTest extends TestCase
{
    /**
     * Data provider for initializeTest1
     *
     * @return array<string, array<string, int>>
     */
    public static function initializeTest1DataProvider() : array
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
     * Data provider for initializeTest2
     *
     * @return array<string, array<string, int>>
     */
    public static function initializeTest2DataProvider() : array
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
     * Data provider for initializeTest3
     *
     * @return array<string, array<string, int>>
     */
    public static function initializeTest3DataProvider() : array
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
    public function constructTest1() : void
    {
        //expect
        $this->expectNotToPerformAssertions();

        //when
        new XavierNormal();
    }

    #[Test]
    #[TestDox('The result matrix has correct shape')]
    #[DataProvider('initializeTest1DataProvider')]
    public function initializeTest1(int $fanIn, int $fanOut) : void
    {
        //given
        $w = new XavierNormal()->initialize(fanIn: $fanIn, fanOut: $fanOut);

        //when
        $shape = $w->shape();

        //then
        $this->assertSame([$fanOut, $fanIn], $shape);
    }

    #[Test]
    #[TestDox('The resulting values matches distribution Xavier (normal distribution)')]
    #[DataProvider('initializeTest2DataProvider')]
    public function initializeTest2(int $fanIn, int $fanOut) : void
    {
        //given
        $expectedStd = sqrt(2 / ($fanOut + $fanIn));
        $w = new XavierNormal()->initialize(fanIn: $fanIn, fanOut:  $fanOut);
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
            'Standard deviation does not match Xavier Normal initialization'
        );
    }

    #[Test]
    #[TestDox('An exception is thrown during initialization')]
    #[DataProvider('initializeTest3DataProvider')]
    public function initializeTest3(int $fanIn, int $fanOut) : void
    {
        //expect
        if ($fanIn < 1) {
            $this->expectException(InvalidFanInException::class);
            $this->expectExceptionMessage("Fan in cannot be less than 1, $fanIn given");
        } elseif ($fanOut < 1) {
            $this->expectException(InvalidFanOutException::class);
            $this->expectExceptionMessage("Fan oun cannot be less than 1, $fanOut given");
        } else {
            $this->expectNotToPerformAssertions();
        }

        //when
        new XavierNormal()->initialize(fanIn: $fanIn, fanOut: $fanOut);
    }

    #[Test]
    #[TestDox('String representation is correct')]
    public function toStringTest1() : void
    {
        //when
        $string = (string) new XavierNormal();

        //then
        $this->assertEquals('Xavier Normal', $string);
    }
}
