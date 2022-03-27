<?php

namespace Rubix\ML\Tests;

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
use function Rubix\ML\warn_deprecated;

/**
 * @group Functions
 * @covers \Rubix\ML\argmax
 * @covers \Rubix\ML\argmin
 * @covers \Rubix\ML\array_transpose
 * @covers \Rubix\ML\comb
 * @covers \Rubix\ML\iterator_contains_nan
 * @covers \Rubix\ML\iterator_filter
 * @covers \Rubix\ML\iterator_first
 * @covers \Rubix\ML\iterator_map
 * @covers \Rubix\ML\linspace
 * @covers \Rubix\ML\logsumexp
 * @covers \Rubix\ML\sigmoid
 * @covers \Rubix\ML\warn_deprecated
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
     * @dataProvider argmaxProvider
     *
     * @param float[] $input
     * @param string|int $expected
     */
    public function argmax(array $input, $expected) : void
    {
        $this->assertEquals($expected, argmax($input));
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function argmaxProvider() : Generator
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

    /**
     * @test
     */
    public function argmaxUndefined() : void
    {
        $this->expectException(RuntimeException::class);

        argmax([NAN, NAN, NAN]);
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
     * @dataProvider sigmoidProvider
     *
     * @param float $value
     * @param float $expected
     */
    public function sigmoid(float $value, float $expected) : void
    {
        $this->assertEquals($expected, sigmoid($value));
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function sigmoidProvider() : Generator
    {
        yield [2.0, 0.8807970779778823];

        yield [-2.0, 0.11920292202211755];

        yield [0.0, 0.5];

        yield [10.0, 0.9999546021312976];
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
     * @return \Generator<mixed[]>
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
     * @dataProvider linspaceProvider
     *
     * @param float $min
     * @param float $max
     * @param int $n
     * @param list<float> $expected
     */
    public function linspace(float $min, float $max, int $n, array $expected) : void
    {
        $this->assertEquals($expected, linspace($min, $max, $n));
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function linspaceProvider() : Generator
    {
        yield [0.0, 1.0, 5, [
            0.0, 0.25, 0.5, 0.75, 1.0,
        ]];

        yield [-4000.0, 6.0, 8, [
            -4000.0, -3427.714285714286, -2855.4285714285716, -2283.1428571428573,
            -1710.8571428571431, -1138.571428571429, -566.2857142857146, 6.0,
        ]];
    }

    /**
     * @test
     * @dataProvider arrayTransposeProvider
     *
     * @param list<list<float>> $table
     * @param list<list<float>> $expected
     */
    public function arrayTranspose(array $table, array $expected) : void
    {
        $this->assertEquals($expected, array_transpose($table));
    }

    /**
     * @return \Generator<mixed[]>
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
     * @return \Generator<mixed[]>
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

    // Until PHP Unit bug is fixed, this needs to be commented out.
    //
    // /**
    //  * @test
    //  */
    // public function warnDeprecated() : void
    // {
    //     $this->expectDeprecation();

    //     warn_deprecated('full control');
    // }
}
