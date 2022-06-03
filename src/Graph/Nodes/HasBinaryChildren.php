<?php

namespace Rubix\ML\Graph\Nodes;

use Traversable;

/**
 * Binary Node
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface HasBinaryChildren extends BinaryNode
{
    /**
     * Return the left child node.
     *
     * @return \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    public function left() : ?BinaryNode;

    /**
     * Return the right child node.
     *
     * @return \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    public function right() : ?BinaryNode;

    /**
     * Return the children of this node in an iterator.
     *
     * @return \Traversable<\Rubix\ML\Graph\Nodes\BinaryNode>
     */
    public function children() : Traversable;

    /**
     * The balance factor of the node. Negative numbers indicate a lean to the left, positive
     * to the right, and 0 is perfectly balanced.
     *
     * @return int
     */
    public function balance() : int;

    /**
     * Set the left child node.
     *
     * @param \Rubix\ML\Graph\Nodes\BinaryNode|null $node
     */
    public function attachLeft(?BinaryNode $node = null) : void;

    /**
     * Set the right child node.
     *
     * @param \Rubix\ML\Graph\Nodes\BinaryNode|null $node
     */
    public function attachRight(?BinaryNode $node = null) : void;
}
