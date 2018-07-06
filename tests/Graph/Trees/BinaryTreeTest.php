<?php

namespace Rubix\Tests\Graph\Trees;

use Rubix\ML\Graph\Trees\Tree;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Trees\BinaryTree;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class BinaryTreeTest extends TestCase
{
    protected $tree;

    public function setUp()
    {
        $this->tree = new BinaryTree();
    }

    public function test_build_tree()
    {
        $this->assertInstanceOf(BinaryTree::class, $this->tree);
        $this->assertInstanceOf(Tree::class, $this->tree);
    }
}
