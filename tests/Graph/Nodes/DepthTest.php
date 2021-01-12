<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Depth;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

/**
 * @group Nodes
 * @covers \Rubix\ML\Graph\Nodes\Depth
 */
class DepthTest extends TestCase
{
    protected const SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
        [-0.01, 0.1, -7],
    ];

    protected const DEPTH = 8;

    protected const C = 8.207392357589622;

    /**
     * @var \Rubix\ML\Graph\Nodes\Depth
     */
    protected $node;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->node = new Depth(self::C);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Depth::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    /**
     * @test
     */
    public function c() : void
    {
        $this->assertEquals(3.748880484475505, Depth::c(10));
        $this->assertEquals(8.364671030072245, Depth::c(100));
        $this->assertEquals(12.969940887100174, Depth::c(1000));
        $this->assertEquals(17.575112063754766, Depth::c(10000));
        $this->assertEquals(22.180282259643523, Depth::c(100000));
    }

    /**
     * @test
     */
    public function terminate() : void
    {
        $dataset = Unlabeled::quick(self::SAMPLES);

        $node = Depth::terminate($dataset, self::DEPTH);

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
