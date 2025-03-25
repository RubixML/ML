<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Initializers\He;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\NeuralNet\Initializers\Xavier\XavierUniform;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanInException;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanOutException;

#[Group('Initializers')]
#[CoversClass(XavierUniform::class)]
final class XavierUniformTest extends TestCase
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
        new XavierUniform();
    }

    #[Test]
    #[TestDox('The result matrix has correct shape')]
    #[DataProvider('initializeTest1DataProvider')]
    public function initializeTest1(int $fanIn, int $fanOut) : void
    {
        //given
        $w = new XavierUniform()->initialize(fanIn: $fanIn, fanOut: $fanOut);

        //when
        $shape = $w->shape();

        //then
        $this->assertSame([$fanOut, $fanIn], $shape);
    }

    #[Test]
    #[TestDox('The resulting values matches distribution Xavier (uniform distribution)')]
    #[DataProvider('initializeTest2DataProvider')]
    public function initializeTest2(int $fanIn, int $fanOut) : void
    {
        //given
        $limit = sqrt(6 / ($fanOut + $fanIn));

        //when
        $w = new XavierUniform()->initialize(fanIn: $fanIn, fanOut:  $fanOut);
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
        new XavierUniform()->initialize(fanIn: $fanIn, fanOut: $fanOut);
    }

    #[Test]
    #[TestDox('String representation is correct')]
    public function toStringTest1() : void
    {
        //when
        $string = (string) new XavierUniform();

        //then
        $this->assertEquals('Xavier Uniform', $string);
    }
}
