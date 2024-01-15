<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Nodes\Isolator;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Nodes
 * @covers \Rubix\ML\Graph\Nodes\Isolator
 */
class IsolatorTest extends TestCase
{
    protected const COLUMN = 1;

    protected const VALUE = 3.0;

    protected const SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
    ];

    /**
     * @var Isolator
     */
    protected $node;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $subsets = [
            Unlabeled::quick([self::SAMPLES[0]]),
            Unlabeled::quick([self::SAMPLES[1]]),
        ];

        $this->node = new Isolator(self::COLUMN, self::VALUE, $subsets);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Isolator::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    /**
     * @test
     */
    public function split() : void
    {
        $dataset = Unlabeled::quick(self::SAMPLES);

        $node = Isolator::split($dataset);

        $this->assertInstanceOf(Isolator::class, $node);
    }

    /**
     * @test
     */
    public function column() : void
    {
        $this->assertSame(self::COLUMN, $this->node->column());
    }

    /**
     * @test
     */
    public function value() : void
    {
        $this->assertSame(self::VALUE, $this->node->value());
    }

    /**
     * @test
     */
    public function cleanup() : void
    {
        $subsets = $this->node->subsets();

        $this->assertIsArray($subsets);
        $this->assertCount(2, $subsets);

        $this->node->cleanup();

        $this->expectException(RuntimeException::class);

        $this->node->subsets();
    }
}
