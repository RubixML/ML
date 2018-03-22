<?php

use Rubix\Engine\BinaryNode;
use PHPUnit\Framework\TestCase;

class BinaryNodeTest extends TestCase
{
    protected $node;

    public function setUp()
    {
        $this->node = new BinaryNode(40, ['coolness_factor' => 'medium']);
    }

    public function test_create_binary_node()
    {
        $this->assertTrue($this->node instanceof BinaryNode);
        $this->assertEquals('medium', $this->node->coolness_factor);

        $this->assertEquals(1, $this->node->height());
        $this->assertNull($this->node->left());
        $this->assertNull($this->node->right());
        $this->assertTrue($this->node->isLeaf());
    }

    public function test_get_node_value()
    {
        $this->assertEquals(40, $this->node->value());
    }

    public function test_attach_left_child()
    {
        $this->assertNull($this->node->left());
        $this->assertTrue($this->node->isLeaf());

        $node = $this->node->attachLeft(new BinaryNode(16, ['coolness_factor' => 'low']));

        $this->assertNotNull($node->left());
        $this->assertTrue($node->left() instanceof BinaryNode);
        $this->assertEquals(2, $node->height());
        $this->assertEquals(1, $node->left()->height());
        $this->assertFalse($node->isLeaf());
    }

    public function test_attach_right_child()
    {
        $this->assertNull($this->node->right());
        $this->assertTrue($this->node->isLeaf());

        $node = $this->node->attachRight(new BinaryNode(19, ['coolness_factor' => 'high']));

        $this->assertNotNull($node->right());
        $this->assertTrue($node->right() instanceof BinaryNode);
        $this->assertEquals(2, $node->height());
        $this->assertEquals(1, $node->right()->height());
        $this->assertFalse($node->isLeaf());
    }

    public function test_detach_left_child()
    {
        $this->node->attachLeft(new BinaryNode(9, ['coolness_factor' => 'high']));

        $this->assertNotNull($this->node->left());
        $this->assertTrue($this->node->left() instanceof BinaryNode);
        $this->assertEquals('high', $this->node->left()->coolness_factor);
        $this->assertEquals($this->node, $this->node->left()->parent());

        $this->node->detachLeft();

        $this->assertNull($this->node->left());
    }

    public function test_detach_right_child()
    {
        $this->node->attachRight(new BinaryNode(12, ['coolness_factor' => 'low']));

        $this->assertNotNull($this->node->right());
        $this->assertTrue($this->node->right() instanceof BinaryNode);
        $this->assertEquals('low', $this->node->right()->coolness_factor);
        $this->assertEquals($this->node, $this->node->right()->parent());

        $this->node->detachRight();

        $this->assertNull($this->node->right());
    }
}
