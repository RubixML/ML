<?php

namespace Rubix\ML\Tests;

use PHPUnit\Framework\TestCase;
use Generator;

use function Rubix\ML\argmin;
use function Rubix\ML\argmax;
use function Rubix\ML\logsumexp;
use function Rubix\ML\comb;
use function Rubix\ML\array_transpose;
use function Rubix\ML\iterator_first;
use function Rubix\ML\iterator_map;
use function Rubix\ML\iterator_filter;
use function Rubix\ML\iterator_contains_nan;
use function Rubix\ML\warn_deprecated;

/**
 * @group Functions
 */
class FunctionsTest extends TestCase
{
    /**
     * Constant used to see the random number generator.
     *
     * @var int
     */
    protected const RANDOM_SEED = 0;

    /**
     * @test
     */
    public function argmin() : void
    {
        $value = argmin(['yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0]);

        $this->assertEquals('maybe', $value);
    }

    /**
     * @test
     */
    public function argmax() : void
    {
        $value = argmax(['yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0]);

        $this->assertEquals('yes', $value);
    }

    /**
     * @test
     */
    public function logsumexp() : void
    {
        $value = logsumexp([0.5, 0.4, 0.9, 1.0, 0.2, 0.9, 0.1, 0.5, 0.7]);

        $this->assertEquals(2.8194175400311074, $value);
    }

    /**
     * @test
     * @dataProvider combProvider
     *
     * @param int $n
     * @param int $k
     * @param int $expected
     */
    public function comb(int $n, int $k, int $expected) : void
    {
        $this->assertEquals($expected, comb($n, $k));
    }

    /**
     * @return \Generator<array>
     */
    public function combProvider() : Generator
    {
        yield [1, 1, 1];

        yield [2, 1, 2];

        yield [8, 3, 56];

        yield [10, 6, 210];
    }

    /**
     * @test
     * @dataProvider arrayTransposeProvider
     *
     * @param array[] $table
     * @param array[] $expected
     */
    public function arrayTranspose(array $table, array $expected) : void
    {
        $this->assertEquals($expected, array_transpose($table));
    }

    /**
     * @return \Generator<array>
     */
    public function arrayTransposeProvider() : Generator
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

    /**
     * @test
     */
    public function iteratorFirst() : void
    {
        $element = iterator_first(['first', 'last']);

        $this->assertEquals('first', $element);
    }

    /**
     * @test
     */
    public function iteratorMap() : void
    {
        $doubleIt = function ($value) {
            return $value * 2;
        };

        $values = iterator_map([3, 6, 9], $doubleIt);

        $expected = [6, 12, 18];

        $this->assertEquals($expected, iterator_to_array($values));
    }

    /**
     * @test
     */
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
     * @test
     * @dataProvider iteratorContainsNanProvider
     *
     * @param mixed[] $values
     * @param bool $expected
     */
    public function iteratorContainsNan(array $values, bool $expected) : void
    {
        $this->assertEquals($expected, iterator_contains_nan($values));
    }

    /**
     * @return \Generator<array>
     */
    public function iteratorContainsNanProvider() : Generator
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
     * @test
     */
    public function warnDeprecated() : void
    {
        $this->expectDeprecation();

        warn_deprecated('deprecated');
    }
}
