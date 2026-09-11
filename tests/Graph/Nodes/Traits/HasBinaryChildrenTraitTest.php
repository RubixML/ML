<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Graph\Nodes\Traits;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Split;
use PHPUnit\Framework\TestCase;

#[Group('Nodes')]
#[CoversClass(Split::class)]
class HasBinaryChildrenTraitTest extends TestCase
{
    protected Split $node;

    protected function setUp() : void
    {
        $subsets = [
            Labeled::quick(),
            Labeled::quick(),
        ];

        $this->node = new Split(
            column: 0,
            value: 0.0,
            subsets: $subsets,
            impurity: 0.0,
            n: 0
        );
    }

    #[Test]
    public function heightOfSingleNode() : void
    {
        $this->assertSame(1, $this->node->height());
    }

    #[Test]
    public function leftIsNullByDefault() : void
    {
        $this->assertNull($this->node->left());
    }

    #[Test]
    public function rightIsNullByDefault() : void
    {
        $this->assertNull($this->node->right());
    }

    #[Test]
    public function childrenIsEmptyByDefault() : void
    {
        $children = iterator_to_array($this->node->children());

        $this->assertSame([], $children);
    }

    #[Test]
    public function attachLeft() : void
    {
        $child = new Split(
            column: 1,
            value: 5.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );

        $this->node->attachLeft($child);

        $this->assertSame($child, $this->node->left());
    }

    #[Test]
    public function attachRight() : void
    {
        $child = new Split(
            column: 2,
            value: -1.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );

        $this->node->attachRight($child);

        $this->assertSame($child, $this->node->right());
    }

    #[Test]
    public function childrenYieldsLeftThenRight() : void
    {
        $left = new Split(
            column: 1,
            value: 5.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );
        $right = new Split(
            column: 2,
            value: -1.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );

        $this->node->attachLeft($left);
        $this->node->attachRight($right);

        $children = iterator_to_array($this->node->children());

        $this->assertCount(2, $children);
        $this->assertSame($left, $children[0]);
        $this->assertSame($right, $children[1]);
    }

    #[Test]
    public function heightIncreasesWithChildren() : void
    {
        $left = new Split(
            column: 1,
            value: 5.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );
        $right = new Split(
            column: 2,
            value: -1.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );

        $this->node->attachLeft($left);
        $this->node->attachRight($right);

        $this->assertSame(2, $this->node->height());
    }

    #[Test]
    public function balanceWithEqualChildren() : void
    {
        $left = new Split(
            column: 1,
            value: 5.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );
        $right = new Split(
            column: 2,
            value: -1.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );

        $this->node->attachLeft($left);
        $this->node->attachRight($right);

        $this->assertSame(0, $this->node->balance());
    }

    #[Test]
    public function balanceWithLeftChildOnly() : void
    {
        $left = new Split(
            column: 1,
            value: 5.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );

        $this->node->attachLeft($left);

        $this->assertSame(-1, $this->node->balance());
    }

    #[Test]
    public function balanceWithRightChildOnly() : void
    {
        $right = new Split(
            column: 2,
            value: -1.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );

        $this->node->attachRight($right);

        $this->assertSame(1, $this->node->balance());
    }

    #[Test]
    public function detachLeft() : void
    {
        $child = new Split(
            column: 1,
            value: 5.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );

        $this->node->attachLeft($child);
        $this->node->attachLeft(null);

        $this->assertNull($this->node->left());
    }

    #[Test]
    public function detachRight() : void
    {
        $child = new Split(
            column: 2,
            value: -1.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );

        $this->node->attachRight($child);
        $this->node->attachRight(null);

        $this->assertNull($this->node->right());
    }

    #[Test]
    public function heightReturnsToOneAfterDetach() : void
    {
        $left = new Split(
            column: 1,
            value: 5.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );
        $right = new Split(
            column: 2,
            value: -1.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 0.0,
            n: 0
        );

        $this->node->attachLeft($left);
        $this->node->attachRight($right);

        $this->assertSame(2, $this->node->height());

        $this->node->attachLeft(null);
        $this->node->attachRight(null);

        $this->assertSame(1, $this->node->height());
    }
}
