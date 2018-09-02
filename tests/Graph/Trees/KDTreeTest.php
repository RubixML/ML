<?php

namespace Rubix\Tests\Graph\Trees;

use Rubix\ML\Graph\Trees\Tree;
use Rubix\ML\Graph\Trees\KDTree;
use PHPUnit\Framework\TestCase;

class KDTreeTest extends TestCase
{
    protected $tree;

    public function setUp()
    {
        $this->tree = new KDTree();
    }

    public function test_build_tree()
    {
        $this->assertInstanceOf(KDTree::class, $this->tree);
        $this->assertInstanceOf(Tree::class, $this->tree);
    }
}
