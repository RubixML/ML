<?php

namespace Rubix\ML\Tests;

use PHPUnit\Framework\TestCase;
use Generator;

use function Rubix\ML\argmin;
use function Rubix\ML\argmax;
use function Rubix\ML\logsumexp;
use function Rubix\ML\array_transpose;

class FunctionsTest extends TestCase
{
    public function test_argmin() : void
    {
        $value = argmin(['yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0]);

        $this->assertEquals('maybe', $value);
    }

    public function test_argmax() : void
    {
        $value = argmax(['yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0]);

        $this->assertEquals('yes', $value);
    }

    public function test_logsumexp() : void
    {
        $value = logsumexp([0.5, 0.4, 0.9, 1.0, 0.2, 0.9, 0.1, 0.5, 0.7]);

        $this->assertEquals(2.8194175400311074, $value);
    }

    /**
     * @param array[] $table
     * @param array[] $expected
     *
     * @dataProvider transpose_provider
     */
    public function test_transpose(array $table, array $expected) : void
    {
        $this->assertEquals($expected, array_transpose($table));
    }

    /**
     * @return \Generator<array>
     */
    public function transpose_provider() : Generator
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
}
