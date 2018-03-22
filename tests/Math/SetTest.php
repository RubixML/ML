<?php

use Rubix\Engine\Math\Set;
use PHPUnit\Framework\TestCase;

class SetTest extends TestCase
{
    protected $set;

    public function setUp()
    {
        $this->set = new Set([16, 17, 59, 48, 27, 36, 14, 12, 11, 18, 36]);
    }

    public function test_build_set()
    {
        $this->assertTrue($this->set instanceof Set);
        $this->assertEquals([11, 12, 14, 16, 17, 18, 27, 36, 48, 59], $this->set->values());
    }

    public function test_get_union()
    {
        $set = new Set([15, 49, 16, 9, 11]);

        $set = $this->set->union($set);

        $this->assertEquals(13, $set->cardinality());
        $this->assertEquals([9, 11, 12, 14, 15, 16, 17, 18, 27, 36, 48, 49, 59], $set->values());
    }

    public function test_get_intersection()
    {
        $set = new Set([15, 49, 16, 9, 11]);

        $set = $this->set->intersection($set);

        $this->assertEquals(2, $set->cardinality());
        $this->assertEquals([11, 16], $set->values());
    }

    public function test_get_difference()
    {
        $set = new Set([15, 49, 16, 9, 11]);

        $set = $this->set->difference($set);

        $this->assertEquals(8, $set->cardinality());
        $this->assertEquals([12, 14, 17, 18, 27, 36, 48, 59], $set->values());
    }

    public function test_get_product()
    {
        $set = new Set(['a', 'b', 'c']);

        $answer = [
            ['a', 'x'],
            ['a', 'y'],
            ['a', 'z'],
            ['b', 'x'],
            ['b', 'y'],
            ['b', 'z'],
            ['c', 'x'],
            ['c', 'y'],
            ['c', 'z'],
        ];

        $this->assertEquals($answer, array_map(function ($set) {
            return $set->values();
        }, $set->product(new Set(['x', 'y', 'z']))));
    }

    public function test_get_power_set()
    {
        $set = new Set(['a', 'b', 'c']);

        $answer = [
            [],
            ['a'],
            ['b'],
            ['a', 'b'],
            ['c'],
            ['a', 'c'],
            ['b', 'c'],
            ['a','b', 'c'],
        ];

        $this->assertEquals($answer, array_map(function ($set) {
            return $set->values();
        }, $set->power()));
    }

    public function test_get_cardinality()
    {
        $this->assertEquals(10, $this->set->cardinality());
    }
}
