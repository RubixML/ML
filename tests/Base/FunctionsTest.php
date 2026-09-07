<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Base;

use PHPUnit\Framework\Attributes\CoversFunction;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use Generator;

use function Rubix\ML\argmin;
use function Rubix\ML\argmax;
use function Rubix\ML\logsumexp;
use function Rubix\ML\minmax;
use function Rubix\ML\sigmoid;
use function Rubix\ML\comb;
use function Rubix\ML\linspace;
use function Rubix\ML\array_pack;
use function Rubix\ML\array_transpose;
use function Rubix\ML\iterator_first;
use function Rubix\ML\iterator_map;
use function Rubix\ML\iterator_filter;
use function Rubix\ML\iterator_contains_nan;
use function Rubix\ML\warn;
use function Rubix\ML\warn_deprecated;
use function is_infinite;

#[Group('Functions')]
#[CoversFunction('\Rubix\ML\argmax')]
#[CoversFunction('\Rubix\ML\argmin')]
#[CoversFunction('\Rubix\ML\array_pack')]
#[CoversFunction('\Rubix\ML\array_transpose')]
#[CoversFunction('\Rubix\ML\comb')]
#[CoversFunction('\Rubix\ML\iterator_contains_nan')]
#[CoversFunction('\Rubix\ML\iterator_filter')]
#[CoversFunction('\Rubix\ML\iterator_first')]
#[CoversFunction('\Rubix\ML\iterator_map')]
#[CoversFunction('\Rubix\ML\linspace')]
#[CoversFunction('\Rubix\ML\logsumexp')]
#[CoversFunction('\Rubix\ML\minmax')]
#[CoversFunction('\Rubix\ML\sigmoid')]
#[CoversFunction('\Rubix\ML\warn')]
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

    public static function arrayPackProvider() : Generator
    {
        yield [
            [
                'row_a' => ['x' => 1.0, 'y' => 2.0],
                'row_b' => ['x' => 3.0, 'y' => 4.0],
                'row_c' => [5.0, 'y' => 7.0],
            ],
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 7.0],
            ],
        ];

        yield [
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ],
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ],
        ];
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

    /**
     * @return Generator<mixed[]>
     */
    public static function sigmoidProvider() : Generator
    {
        yield [2.0, 0.8807970779778823];

        yield [-2.0, 0.11920292202211755];

        yield [0.0, 0.5];

        yield [10.0, 0.9999546021312976];
    }

    /**
     * @return Generator<mixed[]>
     */
    public static function minmaxProvider() : Generator
    {
        yield [[4.2], 4.2, 4.2];

        yield [[1, 2, 3, 4, 5], 1, 5];

        yield [[-3.5, 0.1, 2.7, -3.5], -3.5, 2.7];

        yield [[42, 'yes' => 0.8, 7], 0.8, 42];
    }

    #[Test]
    public function logsumexp() : void
    {
        $value = logsumexp([0.5, 0.4, 0.9, 1.0, 0.2, 0.9, 0.1, 0.5, 0.7]);

        $this->assertEquals(2.8194175400311074, $value);

        $extreme = logsumexp([-1000.0, -1001.0]);

        $this->assertEquals(-999.6867383124818, $extreme);

        $this->assertFalse(is_infinite($extreme));

        $this->assertEquals(-INF, logsumexp([-INF, -INF]));
    }

    #[Test]
    public function argmin() : void
    {
        $value = argmin(['yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0]);

        $this->assertEquals('maybe', $value);
    }

    /**
     * @param float[] $input
     * @param string|int $expected
     */
    #[Test]
    #[DataProvider('argmaxProvider')]
    public function argmax(array $input, mixed $expected) : void
    {
        $this->assertEquals($expected, argmax($input));
    }

    #[Test]
    public function argmaxUndefined() : void
    {
        $this->expectException(RuntimeException::class);

        argmax([NAN, NAN, NAN]);
    }

    /**
     * @param (float|int)[] $input
     * @param float|int $min
     * @param float|int $max
     */
    #[Test]
    #[DataProvider('minmaxProvider')]
    public function minmax(array $input, $min, $max) : void
    {
        $this->assertEquals([$min, $max], minmax($input));
    }

    /**
     * @param float $value
     * @param float $expected
     */
    #[Test]
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
    #[Test]
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
    #[Test]
    #[DataProvider('linspaceProvider')]
    public function linspace(float $min, float $max, int $n, array $expected) : void
    {
        $this->assertEquals($expected, linspace($min, $max, $n));
    }

    /**
     * @param list<list<float>> $table
     * @param list<list<float>> $expected
     */
    #[Test]
    #[DataProvider('arrayTransposeProvider')]
    public function arrayTranspose(array $table, array $expected) : void
    {
        $this->assertEquals($expected, array_transpose($table));
    }

    /**
     * @param array<array<int|float>> $samples
     * @param array<array<int|float>> $expected
     */
    #[Test]
    #[DataProvider('arrayPackProvider')]
    public function arrayPack(array $samples, array $expected) : void
    {
        $this->assertEquals($expected, array_pack($samples));
    }

    #[Test]
    public function iteratorFirst() : void
    {
        $element = iterator_first(['first', 'last']);

        $this->assertEquals('first', $element);
    }

    #[Test]
    public function iteratorMap() : void
    {
        $doubleIt = function ($value) {
            return $value * 2;
        };

        $values = iterator_map([3, 6, 9], $doubleIt);

        $expected = [6, 12, 18];

        $this->assertEquals($expected, iterator_to_array($values));
    }

    #[Test]
    public function iteratorFilter() : void
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
    #[Test]
    #[DataProvider('iteratorContainsNanProvider')]
    public function iteratorContainsNan(array $values, bool $expected) : void
    {
        $this->assertEquals($expected, iterator_contains_nan($values));
    }

    #[Test]
    public function argminUndefinedOnEmptySet() : void
    {
        $this->expectException(InvalidArgumentException::class);

        argmin([]);
    }

    #[Test]
    public function argminUndefinedOnNanSet() : void
    {
        $this->expectException(RuntimeException::class);

        argmin([NAN, NAN]);
    }

    #[Test]
    public function minmaxUndefinedOnEmptySet() : void
    {
        $this->expectException(InvalidArgumentException::class);

        minmax([]);
    }

    #[Test]
    public function argmaxUndefinedOnEmptySet() : void
    {
        $this->expectException(InvalidArgumentException::class);

        argmax([]);
    }

    #[Test]
    public function logsumexpUndefinedOnEmptySet() : void
    {
        $this->expectException(InvalidArgumentException::class);

        logsumexp([]);
    }

    #[Test]
    public function linspaceRejectsMinGreaterThanMax() : void
    {
        $this->expectException(InvalidArgumentException::class);

        linspace(1.0, 0.0, 5);
    }

    #[Test]
    public function linspaceRejectsFewerThanTwoElements() : void
    {
        $this->expectException(InvalidArgumentException::class);

        linspace(0.0, 1.0, 1);
    }

    #[Test]
    public function iteratorFirstUndefinedOnEmptyIterator() : void
    {
        $this->expectException(RuntimeException::class);

        iterator_first([]);
    }

    #[Test]
    public function warnEmitsUserWarning() : void
    {
        $level = null;

        set_error_handler(function (int $errno, string $errstr) use (&$level) : bool {
            $level = $errno;

            return true;
        });

        try {
            warn('A user warning.');
        } finally {
            restore_error_handler();
        }

        $this->assertSame(E_USER_WARNING, $level);
    }

    #[Test]
    public function warnDeprecatedEmitsUserDeprecation() : void
    {
        $level = null;

        set_error_handler(function (int $errno, string $errstr) use (&$level) : bool {
            $level = $errno;

            return true;
        });

        try {
            warn_deprecated('A deprecation warning.');
        } finally {
            restore_error_handler();
        }

        $this->assertSame(E_USER_DEPRECATED, $level);
    }

    #[Test]
    public function logsumexpWithInfinity() : void
    {
        $this->assertSame(INF, logsumexp([INF, 1.0]));
    }

    #[Test]
    public function combWithKGreaterThanN() : void
    {
        $this->assertSame(0, comb(3, 5));
    }

    #[Test]
    public function arrayTransposeOnEmptyTable() : void
    {
        $this->assertSame([], array_transpose([]));
    }

    #[Test]
    public function iteratorMapWithGeneratorInput() : void
    {
        $values = iterator_map((function () {
            yield 1;
            yield 2;
            yield 3;
        })(), function ($value) {
            return $value * 2;
        });

        $this->assertEquals([2, 4, 6], iterator_to_array($values));
    }

    #[Test]
    public function iteratorFilterWithGeneratorInput() : void
    {
        $values = iterator_filter((function () {
            yield 1;
            yield 2;
            yield 3;
        })(), function ($value) {
            return $value >= 2;
        });

        $this->assertEquals([2, 3], iterator_to_array($values));
    }

    #[Test]
    public function arrayPackOnEmptySamples() : void
    {
        $this->assertSame([], array_pack([]));
    }
}
