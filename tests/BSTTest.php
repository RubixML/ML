<?php

use Rubix\Engine\BST;
use Rubix\Engine\BinaryNode;
use PHPUnit\Framework\TestCase;

class BSTTest extends TestCase
{
    protected $tree;

    public function setUp()
    {
        $this->tree = new BST([20, 6, 15, 47, 32, 49, 58, 27, 99, 42, 12, 16,
            75, 72, 2, 77, 79, 88, 83]);
    }

    public function test_build_bst()
    {
        $this->assertTrue($this->tree instanceof BST);
        $this->assertEquals(19, $this->tree->size());
    }

    public function test_insert_value()
    {
        $this->assertEquals(19, $this->tree->count());

        $node = $this->tree->insert(14, ['coolness_factor' => 'low']);

        $this->assertEquals(20, $this->tree->count());
        $this->assertTrue($node instanceof BinaryNode);
        $this->assertEquals(14, $node->value());
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

        $this->assertEquals([15, 16, 20, 27, 32, 42, 47], $range);
    }

    public function test_find_bad_range()
    {
        $this->expectException(InvalidArgumentException::class);

        $range = $this->tree->findRange(47, 15);
    }

    public function test_get_all_values()
    {
        $values = $this->tree->all();

        $this->assertEquals([2, 6, 12, 15, 16, 20, 27, 32, 42, 47, 49, 58, 72,
            75, 77, 79, 83, 88, 99], $values);

        $this->assertEquals(19, count($values));
    }

    public function test_get_in_order_successor()
    {
        $this->assertEquals(47, $this->tree->successor($this->tree->find(42))->value());
        $this->assertEquals(49, $this->tree->successor($this->tree->find(47))->value());
        $this->assertEquals(58, $this->tree->successor($this->tree->find(49))->value());
        $this->assertEquals(72, $this->tree->successor($this->tree->find(58))->value());
    }

    public function test_min_value()
    {
        $this->assertEquals(2, $this->tree->min());

        $this->tree->delete(2);

        $this->assertEquals(6, $this->tree->min());
    }

    public function test_max_value()
    {
        $this->assertEquals(99, $this->tree->max());

        $this->tree->delete(99);

        $this->assertEquals(88, $this->tree->max());
    }

    public function test_delete()
    {
        $this->assertEquals([2, 6, 12, 15, 16, 20, 27, 32, 42, 47, 49, 58, 72, 75,
            77, 79, 83, 88, 99], $this->tree->all());

        $this->assertEquals(19, $this->tree->size());

        $this->tree->delete(88);

        $this->assertEquals([2, 6, 12, 15, 16, 20, 27, 32, 42, 47, 49, 58, 72, 75,
            77, 79, 83, 99], $this->tree->all());

        $this->assertEquals(18, $this->tree->size());
    }
}
