<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Helpers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Classifiers\KNearestNeighbors;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Helpers')]
#[CoversClass(Params::class)]
class ParamsTest extends TestCase
{
    public static function stringifyProvider() : Generator
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
            '0: K Nearest Neighbors (k: 5, weighted: false, kernel: Euclidean), 1: 1, 2: 0.8',
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

    protected function setUp() : void
    {
        ini_set('precision', '14');
    }

    public function testInts() : void
    {
        $values = Params::ints(min: 0, max: 100, n: 5);

        $this->assertContainsOnlyInt($values);

        $this->assertEquals(array_unique($values), $values);

        foreach ($values as $value) {
            $this->assertThat(
                $value,
                $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100))
            );
        }
    }

    public function testFloats() : void
    {
        $values = Params::floats(min: 0.0, max: 100.0, n: 5);

        $this->assertContainsOnlyFloat($values);

        foreach ($values as $value) {
            $this->assertThat(
                $value,
                $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100))
            );
        }
    }

    public function testGrid() : void
    {
        $values = Params::grid(min: 0, max: 100, n: 5);

        $this->assertEquals(range(0, 100, 25), $values);
    }

    /**
     * @param array $params
     * @param string $separator
     * @param string $expected
     */
    #[DataProvider('stringifyProvider')]
    public function stringify(array $params, string $separator, string $expected) : void
    {
        $this->assertEquals($expected, Params::stringify(params: $params, separator: $separator));
    }

    public function testSortName() : void
    {
        $this->assertEquals('KNearestNeighbors', Params::shortName(KNearestNeighbors::class));
    }
}
