<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Initializers\Constant;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\NeuralNet\Initializers\Constant\Constant;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanInException;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanOutException;

#[Group('Initializers')]
#[CoversClass(Constant::class)]
class ConstantTest extends TestCase
{
    /**
     * DataProvider for constructTest1
     *
     * @return array<string, array<string, float>>
     */
    public static function constructTest1DataProvider() : array
    {
        return [
            'negative value' => [
                'value' => -3.4,
            ],
            'zero value' => [
                'value' => 0.0,
            ],
            'positive value' => [
                'value' => 0.3,
            ],
        ];
    }

    /**
     * DataProvider for initializeTest1
     *
     * @return array<string, array<string, int<1, max>>>
     */
    public static function initializeTest1DataProvider() : array
    {
        return [
            'fanIn and fanOut being equal' => [
                'fanIn' => 3,
                'fanOut' => 3,
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
     * Data provider for initializeTest3
     *
     * @return array<string, array<string, int<1, max>>>
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
    #[TestDox('The initializer object os created correctly')]
    #[DataProvider('constructTest1DataProvider')]
    public function constructTest1(float $value) : void
    {
        //expect
        $this->expectNotToPerformAssertions();

        //when
        new Constant($value);
    }

    #[Test]
    #[TestDox('The result matrix has correct shape')]
    #[DataProvider('initializeTest1DataProvider')]
    public function initializeTest1(int $fanIn, int $fanOut) : void
    {
        //given
        $w = new Constant(4.8)->initialize(fanIn: $fanIn, fanOut: $fanOut);

        //when
        $shape = $w->shape();

        //then
        $this->assertSame([$fanOut, $fanIn], $shape);
    }

    #[Test]
    #[TestDox('All elements correspond to a single value')]
    public function initializeTest2() : void
    {
        //given
        $w = new Constant(4.5)->initialize(3, 4);

        //when
        $values = $w->toArray();

        //then
        $this->assertEquals([4.5], array_unique(array_merge(...$values)));
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
        new Constant()->initialize(fanIn: $fanIn, fanOut: $fanOut);
    }
}
