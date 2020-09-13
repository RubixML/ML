<?php

namespace Rubix\ML\Tests;

use PHPUnit\Framework\TestCase;
use Generator;

use function Rubix\ML\argmin;
use function Rubix\ML\argmax;
use function Rubix\ML\logsumexp;
use function Rubix\ML\comb;
use function Rubix\ML\array_transpose;
use function Rubix\ML\warn_deprecated;

/**
 * @group Functions
 */
class FunctionsTest extends TestCase
{
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
    public function warnDeprecated() : void
    {
        $this->expectDeprecation();

        warn_deprecated('deprecated');
    }
}
