<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Graph\Nodes\Average;
use Rubix\ML\Graph\Nodes\Decision;
use PHPUnit\Framework\TestCase;

/**
 * @group Nodes
 * @covers \Rubix\ML\Graph\Nodes\Average
 */
class AverageTest extends TestCase
{
    protected const OUTCOME = 44.21;

    protected const IMPURITY = 6.0;

    protected const N = 3;

    /**
     * @var \Rubix\ML\Graph\Nodes\Average
     */
    protected $node;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->node = new Average(self::OUTCOME, self::IMPURITY, self::N);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Average::class, $this->node);
        $this->assertInstanceOf(Outcome::class, $this->node);
        $this->assertInstanceOf(Decision::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    /**
     * @test
     */
    public function outcome() : void
    {
        $this->assertSame(self::OUTCOME, $this->node->outcome());
    }

    /**
     * @test
     */
    public function impurity() : void
    {
        $this->assertSame(self::IMPURITY, $this->node->impurity());
    }

    /**
     * @test
     */
    public function n() : void
    {
        $this->assertSame(self::N, $this->node->n());
    }
}
