<?php

namespace Rubix\ML\Tests\Graph;

use Rubix\ML\Graph\Tree;
use Rubix\ML\Graph\BallTree;
use Rubix\ML\Graph\BinaryTree;
use PHPUnit\Framework\TestCase;

class BallTreeTest extends TestCase
{
    protected $tree;

    public function setUp()
    {
        $this->tree = new BallTree();
    }

    public function test_build_tree()
    {
        $this->assertInstanceOf(BallTree::class, $this->tree);
        $this->assertInstanceOf(BinaryTree::class, $this->tree);
        $this->assertInstanceOf(Tree::class, $this->tree);
    }
}
