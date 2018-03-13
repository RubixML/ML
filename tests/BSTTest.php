<?php

use Rubix\Engine\BST;
use Rubix\Engine\BinaryNode;
use PHPUnit\Framework\TestCase;

class BSTTest extends TestCase
{
    protected $tree;

    public function setUp()
    {
        $this->tree = BST::fromArray([20 => [], 6 => [], 15 => [], 47 => [], 32 => [],
            49 => [], 58 => [], 27 => [], 99 => [], 42 => [], 12 => [], 16 => [],
            75 => [], 72 => [], 2 => [], 77 => [], 79 => [], 88 => [], 83 => []]);
    }

    public function test_build_bst()
    {
        $this->assertTrue($this->tree instanceof BST);
    }

    public function test_insert_value()
    {
        $node = $this->tree->insert(14, ['coolness_factor' => 'low']);

        $this->assertTrue($node instanceof BinaryNode);
        $this->assertEquals(14, $node->value);
        $this->assertEquals('low', $node->coolness_factor);
    }

    public function test_has_value()
    {
        $this->assertTrue($this->tree->has(49));
        $this->assertTrue($this->tree->has(12));
        $this->assertTrue($this->tree->has(88));
        $this->assertFalse($this->tree->has(4));
        $this->assertFalse($this->tree->has(1000));
    }

    public function test_find_range()
    {
        $range = $this->tree->findRange(15, 47);

        $this->assertEquals([15, 16, 20, 27, 32, 42, 47], $range->pluck('value'));
    }

    public function test_find_bad_range()
    {
        $this->expectException(InvalidArgumentException::class);

        $range = $this->tree->findRange(47, 15);
    }

    public function test_get_sorted_path()
    {
        $path = $this->tree->sort();

        $this->assertEquals([2, 6, 12, 15, 16, 20, 27, 32, 42, 47, 49, 58, 72,
            75, 77, 79, 83, 88, 99], $path->pluck('value'));
    }

    public function test_get_in_order_successor()
    {
        $this->assertEquals(6, $this->tree->successor($this->tree->find(2))->value);
        $this->assertEquals(12, $this->tree->successor($this->tree->find(6))->value);
        $this->assertEquals(15, $this->tree->successor($this->tree->find(12))->value);
        $this->assertEquals(16, $this->tree->successor($this->tree->find(15))->value);
        $this->assertEquals(20, $this->tree->successor($this->tree->find(16))->value);
        $this->assertEquals(27, $this->tree->successor($this->tree->find(20))->value);
        $this->assertEquals(32, $this->tree->successor($this->tree->find(27))->value);
        $this->assertEquals(42, $this->tree->successor($this->tree->find(32))->value);
        $this->assertEquals(49, $this->tree->successor($this->tree->find(47))->value);
        $this->assertEquals(58, $this->tree->successor($this->tree->find(49))->value);
        $this->assertEquals(72, $this->tree->successor($this->tree->find(58))->value);
        $this->assertEquals(75, $this->tree->successor($this->tree->find(72))->value);
        $this->assertEquals(77, $this->tree->successor($this->tree->find(75))->value);
        $this->assertEquals(79, $this->tree->successor($this->tree->find(77))->value);
        $this->assertEquals(83, $this->tree->successor($this->tree->find(79))->value);
        $this->assertEquals(88, $this->tree->successor($this->tree->find(83))->value);
        $this->assertEquals(99, $this->tree->successor($this->tree->find(88))->value);
    }

    public function test_min_value()
    {
        $this->assertEquals(2, $this->tree->min()->value);
    }

    public function test_max_value()
    {
        $this->assertEquals(99, $this->tree->max()->value);
    }

    public function test_delete()
    {
        $this->assertEquals([2, 6, 12, 15, 16, 20, 27, 32, 42, 47, 49, 58, 72, 75,
            77, 79, 83, 88, 99], $this->tree->sort()->pluck('value'));

        $this->tree->delete(88);
        $this->tree->delete(20);
        $this->tree->delete(47);
        $this->tree->delete(27);

        $this->assertEquals([2, 6, 12, 15, 16, 32, 42, 49, 58, 72, 75, 77, 79, 83,
            99], $this->tree->sort()->pluck('value'));
    }

    public function test_delete_range()
    {
        $this->tree->deleteRange(16, 80);

        $this->assertEquals([2, 6, 12, 15, 83, 88, 99], $this->tree->sort()->pluck('value'));
    }
}
