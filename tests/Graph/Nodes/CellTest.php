<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Cell;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

/**
 * @group Nodes
 * @covers \Rubix\ML\Graph\Nodes\Cell
 */
class CellTest extends TestCase
{
    protected const SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
        [-0.01, 0.1, -7],
    ];

    protected const DEPTH = 8;

    protected const C = 8.207392357589622;

    /**
     * @var \Rubix\ML\Graph\Nodes\Cell
     */
    protected $node;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->node = new Cell(self::C);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Cell::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Leaf::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    /**
     * @test
     */
    public function terminate() : void
    {
        $dataset = Unlabeled::quick(self::SAMPLES);

        $node = Cell::terminate($dataset, self::DEPTH);

        $this->assertEquals(self::C, $node->depth());
    }

    /**
     * @test
     */
    public function depth() : void
    {
        $this->assertEquals(self::C, $this->node->depth());
    }
}
