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
     * Provides valid values for constructing a Constant initializer.
     *
     * @return array<string, array{value: float}>
     */
    public static function validConstructorValuesProvider() : array
    {
        return [
            'negative constant value' => ['value' => -3.4],
            'zero constant value' => ['value' => 0.0],
            'positive constant value' => ['value' => 0.3],
        ];
    }

    /**
     * Provides valid fanIn and fanOut values to test the shape of the initialized matrix.
     *
     * @return array<string, array{fanIn: int, fanOut: int}>
     */
    public static function validFanInAndFanOutProvider() : array
    {
        return [
            'equal fanIn and fanOut' => ['fanIn' => 3, 'fanOut' => 3],
            'fanIn greater than fanOut' => ['fanIn' => 4, 'fanOut' => 3],
            'fanIn less than fanOut' => ['fanIn' => 3, 'fanOut' => 4],
        ];
    }

    /**
     * Provides invalid fanIn and fanOut values to test exception handling.
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
    #[TestDox('It constructs the initializer with valid values')]
    #[DataProvider('validConstructorValuesProvider')]
    public function testConstructorWithValidValues(float $value) : void
    {
        //expect
        $this->expectNotToPerformAssertions();

        //when
        new Constant($value);
    }

    #[Test]
    #[TestDox('It initializes a matrix with correct shape')]
    #[DataProvider('validFanInAndFanOutProvider')]
    public function testMatrixHasCorrectShape(int $fanIn, int $fanOut) : void
    {
        //given
        $w = new Constant(4.8)->initialize(fanIn: $fanIn, fanOut: $fanOut);

        //when
        $shape = $w->shape();

        //then
        $this->assertSame([$fanOut, $fanIn], $shape);
    }

    #[Test]
    #[TestDox('It initializes a matrix filled with the constant value')]
    public function testMatrixFilledWithConstantValue() : void
    {
        //given
        $w = new Constant(4.5)->initialize(3, 4);

        //when
        $values = $w->toArray();

        //then
        $this->assertEquals([4.5], array_unique(array_merge(...$values)));
    }

    #[Test]
    #[TestDox('It throws an exception when fanIn or fanOut is invalid')]
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
        new Constant()->initialize(fanIn: $fanIn, fanOut: $fanOut);
    }

    #[Test]
    #[TestDox('String representation is correct')]
    public function testReturnsCorrectStringRepresentation() : void
    {
        //when
        $string = (string) new Constant();

        //then
        $this->assertEquals('Constant (value: 0)', $string);
    }
}
