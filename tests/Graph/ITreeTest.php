<?php

namespace Rubix\ML\Tests\Graph;

use Rubix\ML\Graph\Tree;
use Rubix\ML\Graph\ITree;
use Rubix\ML\Graph\BinaryTree;
use Rubix\ML\Graph\Nodes\Cell;
use Rubix\ML\Graph\Nodes\Isolator;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

class ITreeTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    protected $generator;

    protected $tree;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'east' => new Blob([5, -2, -2]),
            'west' => new Blob([0, 5, -3]),
        ], [0.5, 0.5]);

        $this->tree = new ITree();

        srand(self::RANDOM_SEED);
    }

    public function test_build_tree()
    {
        $this->assertInstanceOf(ITree::class, $this->tree);
        $this->assertInstanceOf(BinaryTree::class, $this->tree);
        $this->assertInstanceOf(Tree::class, $this->tree);

        $this->assertNull($this->tree->root());
        $this->assertEquals(0, $this->tree->height());
    }

    public function test_grow_range()
    {
        $this->tree->grow($this->generator->generate(50));

        $this->assertInstanceOf(Isolator::class, $this->tree->root());
        $this->assertInstanceOf(BinaryNode::class, $this->tree->root());

        $this->assertGreaterThan(5, $this->tree->height());

        $sample = $this->generator->generate(1)->row(0);

        $node = $this->tree->search($sample);

        $this->assertInstanceOf(Cell::class, $node);
        $this->assertInstanceOf(BinaryNode::class, $node);
    }
}
