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
        $this->assertEquals(range(0, 100, 25), Params::grid(0, 100, 5));
    }
    
    /**
     * @test
     * @dataProvider stringifyProvider
     *
     * @param mixed[] $params
     * @param string $expected
     */
    public function stringify(array $params, string $expected) : void
    {
        $this->assertEquals($expected, Params::stringify($params));
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
            'learning_rate=0.1 alpha=0.0001 priors=null',
        ];

        yield [
            [
                new KNearestNeighbors(5),
                1.0,
                0.8,
            ],
            '0=KNearestNeighbors(k=5 weighted=true kernel=Euclidean) 1=1 2=0.8',
        ];

        yield [
            [
                1,
                [2, 3, 4],
                5,
            ],
            '0=1 1=[0=2 1=3 2=4] 2=5',
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
