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
interface BinaryNode extends Node
{
    /**
     * Return the children of this node in a generator.
     *
     * @return \Traversable<\Rubix\ML\Graph\Nodes\BinaryNode>
     */
    public function children() : Traversable;

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
     * Recursive function to determine the height of the node in the tree.
     *
     * @return int
     */
    public function height() : int;

    /**
     * The balance factor of the node. Negative numbers indicate a
     * lean to the left, positive to the right, and 0 is perfectly
     * balanced.
     *
     * @return int
     */
    public function balance() : int;

    /**
     * Set the left child node.
     *
     * @param self $node
     */
    public function attachLeft(BinaryNode $node) : void;

    /**
     * Set the right child node.
     *
     * @param self $node
     */
    public function attachRight(BinaryNode $node) : void;

    /**
     * Detach the left child node.
     */
    public function detachLeft() : void;

    /**
     * Detach the right child node.
     */
    public function detachRight() : void;

    /**
     * Is this a leaf node? i.e no children.
     *
     * @return bool
     */
    public function leaf() : bool;
}
