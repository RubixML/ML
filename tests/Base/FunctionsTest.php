<?php

declare(strict_types=1);

namespace Rubix\ML\Tests;

use PHPUnit\Framework\Attributes\CoversFunction;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use Generator;

use function Rubix\ML\argmin;
use function Rubix\ML\argmax;
use function Rubix\ML\logsumexp;
use function Rubix\ML\sigmoid;
use function Rubix\ML\comb;
use function Rubix\ML\linspace;
use function Rubix\ML\array_transpose;
use function Rubix\ML\iterator_first;
use function Rubix\ML\iterator_map;
use function Rubix\ML\iterator_filter;
use function Rubix\ML\iterator_contains_nan;

#[Group('Functions')]
#[CoversFunction('\Rubix\ML\argmax')]
#[CoversFunction('\Rubix\ML\argmin')]
#[CoversFunction('\Rubix\ML\array_transpose')]
#[CoversFunction('\Rubix\ML\comb')]
#[CoversFunction('\Rubix\ML\iterator_contains_nan')]
#[CoversFunction('\Rubix\ML\iterator_filter')]
#[CoversFunction('\Rubix\ML\iterator_first')]
#[CoversFunction('\Rubix\ML\iterator_map')]
#[CoversFunction('\Rubix\ML\linspace')]
#[CoversFunction('\Rubix\ML\logsumexp')]
#[CoversFunction('\Rubix\ML\sigmoid')]
#[CoversFunction('\Rubix\ML\warn_deprecated')]
class FunctionsTest extends TestCase
{
    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    public static function argmaxProvider() : Generator
    {
        yield [
            ['yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0],
            'yes',
        ];

        yield [
            ['yes' => 3.3, 'no' => 3.3, 'maybe' => 3.3],
            'yes',
        ];

        yield [
            ['yes' => 0.8, 'no' => 0.2, 'maybe' => NAN],
            'yes',
        ];
    }

    public static function sigmoidProvider() : Generator
    {
        yield [2.0, 0.8807970779778823];

        yield [-2.0, 0.11920292202211755];

        yield [0.0, 0.5];

        yield [10.0, 0.9999546021312976];
    }

    public static function combProvider() : Generator
    {
        yield [1, 1, 1];

        yield [2, 1, 2];

        yield [8, 3, 56];

        yield [10, 6, 210];
    }

    public static function linspaceProvider() : Generator
    {
        yield [0.0, 1.0, 5, [
            0.0, 0.25, 0.5, 0.75, 1.0,
        ]];

        yield [-4000.0, 6.0, 8, [
            -4000.0, -3427.714285714286, -2855.4285714285716, -2283.1428571428573,
            -1710.8571428571431, -1138.571428571429, -566.2857142857146, 6.0,
        ]];
    }

    public static function arrayTransposeProvider() : Generator
    {
        yield [
            [
                [1, 2, 3, 4],
                [2, 2, 3, 0],
                [3, 3, 0, 0],
                [4, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 2, 3, 4, 0],
                [2, 2, 3, 0, 0],
                [3, 3, 0, 0, 0],
                [4, 0, 0, 0, 0],
            ],
        ];

        yield [
            [
                [1, 0, 0, 0, 0],
            ],
            [
                [1],
                [0],
                [0],
                [0],
                [0],
            ],
        ];
    }

    public static function iteratorContainsNanProvider() : Generator
    {
        yield [
            [0.0, NAN, -5],
            true,
        ];

        yield [
            [0.0, 0.0, 0.0],
            false,
        ];

        yield [
            [1.0, INF, NAN],
            true,
        ];

        yield [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, NAN],
            ],
            true,
        ];

        yield [
            ['NaN', 'NAN'],
            false,
        ];
    }

    public function testArgmin() : void
    {
        $value = argmin(['yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0]);

        $this->assertEquals('maybe', $value);
    }

    /**
     * @param float[] $input
     * @param string|int $expected
     */
    #[DataProvider('argmaxProvider')]
    public function testArgmax(array $input, mixed $expected) : void
    {
        $this->assertEquals($expected, argmax($input));
    }

    public function testArgmaxUndefined() : void
    {
        $this->expectException(RuntimeException::class);

        argmax([NAN, NAN, NAN]);
    }

    public function testLogsumexp() : void
    {
        $value = logsumexp([0.5, 0.4, 0.9, 1.0, 0.2, 0.9, 0.1, 0.5, 0.7]);

        $this->assertEquals(2.8194175400311074, $value);
    }

    /**
     * @param float $value
     * @param float $expected
     */
    #[DataProvider('sigmoidProvider')]
    public function sigmoid(float $value, float $expected) : void
    {
        $this->assertEquals($expected, sigmoid($value));
    }

    /**
     * @param int $n
     * @param int $k
     * @param int $expected
     */
    #[DataProvider('combProvider')]
    public function comb(int $n, int $k, int $expected) : void
    {
        $this->assertEquals($expected, comb($n, $k));
    }

    /**
     * @param float $min
     * @param float $max
     * @param int $n
     * @param list<float> $expected
     */
    #[DataProvider('linspaceProvider')]
    public function linspace(float $min, float $max, int $n, array $expected) : void
    {
        $this->assertEquals($expected, linspace($min, $max, $n));
    }

    /**
     * @param list<list<float>> $table
     * @param list<list<float>> $expected
     */
    #[DataProvider('arrayTransposeProvider')]
    public function arrayTranspose(array $table, array $expected) : void
    {
        $this->assertEquals($expected, array_transpose($table));
    }

    public function testIteratorFirst() : void
    {
        $element = iterator_first(['first', 'last']);

        $this->assertEquals('first', $element);
    }

    public function testIteratorMap() : void
    {
        $doubleIt = function ($value) {
            return $value * 2;
        };

        $values = iterator_map([3, 6, 9], $doubleIt);

        $expected = [6, 12, 18];

        $this->assertEquals($expected, iterator_to_array($values));
    }

    public function testIteratorFilter() : void
    {
        $isPositive = function ($value) {
            return $value >= 0;
        };

        $values = iterator_filter([3, -6, 9], $isPositive);

        $expected = [3, 9];

        $this->assertEquals($expected, iterator_to_array($values));
    }

    /**
     * @param array<array<int|float>|bool> $values
     * @param bool $expected
     */
    #[DataProvider('iteratorContainsNanProvider')]
    public function iteratorContainsNan(array $values, bool $expected) : void
    {
        $this->assertEquals($expected, iterator_contains_nan($values));
    }
}
