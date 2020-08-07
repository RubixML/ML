<?php

namespace Rubix\ML\Tests\Other\Helpers;

use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Classifiers\KNearestNeighbors;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Helpers
 * @covers \Rubix\ML\Other\Helpers\Params
 */
class ParamsTest extends TestCase
{
    /**
     * @test
     */
    public function ints() : void
    {
        $values = Params::ints(0, 100, 5);

        $this->assertContainsOnly('int', $values);

        $this->assertEquals(array_unique($values), $values);

        foreach ($values as $value) {
            $this->assertThat(
                $value,
                $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100))
            );
        }
    }

    /**
     * @test
     */
    public function floats() : void
    {
        $values = Params::floats(0, 100, 5);

        $this->assertContainsOnly('float', $values);

        foreach ($values as $value) {
            $this->assertThat(
                $value,
                $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100))
            );
        }
    }

    /**
     * @test
     */
    public function grid() : void
    {
        $values = Params::grid(0, 100, 5);

        $this->assertEquals(range(0, 100, 25), $values);
    }

    /**
     * @test
     * @dataProvider stringifyProvider
     *
     * @param mixed[] $params
     * @param string $separator
     * @param string $expected
     */
    public function stringify(array $params, string $separator, string $expected) : void
    {
        $this->assertEquals($expected, Params::stringify($params, $separator));
    }

    /**
     * @return \Generator<array>
     */
    public function stringifyProvider() : Generator
    {
        yield [
            [
                'learning_rate' => 0.1,
                'alpha' => 1e-4,
                'priors' => null,
            ],
            ', ',
            'learning_rate: 0.1, alpha: 0.0001, priors: null',
        ];

        yield [
            [
                new KNearestNeighbors(5),
                1.0,
                0.8,
            ],
            ', ',
            '0: K Nearest Neighbors (k: 5, weighted: true, kernel: Euclidean), 1: 1, 2: 0.8',
        ];

        yield [
            [
                1,
                [2, 3, 4],
                5,
            ],
            ' - ',
            '0: 1 - 1: [0: 2, 1: 3, 2: 4] - 2: 5',
        ];
    }

    /**
     * @test
     */
    public function shortName() : void
    {
        $this->assertEquals('KNearestNeighbors', Params::shortName(KNearestNeighbors::class));
    }
}
