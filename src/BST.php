<?php

namespace Rubix\Engine;

use InvalidArgumentException;
use Countable;
use SplStack;

class BST implements Countable
{
    /**
     * The root node of the binary search tree.
     *
     * @var \Rubix\Engine\BinaryNode|null  $root
     */
    protected $root;

    /**
     * The number of nodes stored in the tree.
     *
     * @var int
     */
    protected $size;

    /**
     * Factory method to create a BST from an associative array of values and
     * properties. O(N logV)
     *
     * @param  array  $values
     * @return self
     */
    public static function fromArray(array $values) : self
    {
        $tree = new static();

        $tree->merge($values);

        return $tree;
    }

    /**
     * @return void
     */
    public function __construct()
    {
        $this->root = null;
    }

    /**
     * @return \Rubix\Engine\BinaryNode|null
     */
    public function root() : ?BinaryNode
    {
        return $this->root;
    }

    /**
     * Search the BST for a given value. O(log V)
     *
     * @param  mixed  $value
     * @return bool
     */
    public function has($value) : bool
    {
        return !is_null($this->find($value));
    }

    /**
     * Insert a node into the BST and rebalance. O(log V)
     *
     * @param  mixed  $value
     * @param  array  $properties
     * @return \Rubix\Engine\BinaryNode
     */
    public function insert($value, array $properties = []) : BinaryNode
    {
        $node = new BinaryNode(array_replace($properties, ['value' => $value]));

        if ($this->isEmpty()) {
            $this->root = $node;
        } else {
            $parent = $this->root;

            while ($parent !== null) {
                if ($parent->value > $value) {
                    if (is_null($parent->left())) {
                        $parent->attachLeft($node);
                        break;
                    } else {
                        $parent = $parent->left();
                    }
                } else {
                    if (is_null($parent->right())) {
                        $parent->attachRight($node);
                        break;
                    } else {
                        $parent = $parent->right();
                    }
                }
            }

            while (isset($parent)) {
                $balance = $parent->balance();

                if ($balance > 1 && $node->value < $parent->left()->value) {
                    $this->rotateRight($parent);
                } else if ($balance < -1 && $node->value > $parent->right()->value) {
                    $this->rotateLeft($parent);
                } else if ($balance > 1 && $node->value > $parent->left()->value) {
                    $this->rotateLeft($parent->left());
                    $this->rotateRight($parent);
                } else if ($balance < -1 && $node->value < $parent->right()->value) {
                    $this->rotateRight($parent->right());
                    $this->rotateLeft($parent);
                }

                $parent = $parent->parent();
            }
        }

        $this->size++;

        return $node;
    }

    /**
     * Merge an array of values and properties into the BST. O(N logV)
     *
     * @param  array  $values
     * @return self
     */
    public function merge(array $values) : self
    {
        foreach ($values as $value => $properties) {
            $this->insert($value, $properties);
        }

        return $this;
    }

    /**
     * Find a node with the given value. O(log V)
     *
     * @param  mixed  $value
     * @return \Rubix\Engine\BinaryNode|null
     */
    public function find($value) : ?BinaryNode
    {
        if (!is_numeric($value) && !is_string($value)) {
            throw new InvalidArgumentException('Search value must be numeric or string type, ' . gettype($value) . ' found.');
        }

        $current = $this->root;

        while ($current !== null) {
            if ($current->value === $value) {
                break;
            } else if ($current->value > $value) {
                $current = $current->left();
            } else {
                $current = $current->right();
            }
        }

        return $current;
    }

    /**
     * Find a range of nodes with values between start and end. O(log N)
     *
     * @param  mixed  $start
     * @param  mixed  $end
     * @return \Rubix\Engine\Path
     */
    public function findRange($start, $end) : Path
    {
        if ($start > $end) {
            throw new InvalidArgumentException('Start range value must be less than or equal to end value.');
        }

        $path = new Path();

        $this->_findRange($start, $end, $this->root, $path);

        return $path;
    }

    /**
     * Recursive function to find all nodes with values given a certain range.
     *
     * @param  mixed  $start
     * @param  mixed  $end
     * @param  \Rubix\Engine\BinaryNode  $current
     * @param  \Rubix\Engine\Path  $path
     * @return void
     */
    protected function _findRange($start, $end, BinaryNode $current = null, Path $path) : void
    {
        if (is_null($current)) {
            return;
        }

        if ($current->value > $start) {
            $this->_findRange($start, $end, $current->left(), $path);
        }

        if ($current->value >= $start && $current->value <= $end) {
            $path->append($current);
        }

        if ($current->value < $end) {
            $this->_findRange($start, $end, $current->right(), $path);
        }
    }

    /**
     * Return a path of nodes sorted by value. O(V)
     *
     * @return array
     */
    public function sort() : ?Path
    {
        $stack = new SplStack();
        $path = new Path();

        if ($this->isEmpty()) {
            return $path;
        }

        $current = $this->root;

        while (true) {
            if (isset($current)) {
                $stack->push($current);

                $current = $current->left();
            } else {
                if (!$stack->isEmpty()) {
                    $current = $stack->pop();

                    $path->append($current);

                    $current = $current->right();
                } else {
                    break;
                }
            }
        }

        return $path;
    }

    /**
     * Return the in order successor of a given node or null if given node is max.
     *
     * @param  \Rubix\Engine\BinaryNode  $node
     * @return \Rubix\Engine\BinaryNode|null
     */
    public function successor(BinaryNode $node) : ?BinaryNode
    {
        if (!is_null($node->right())) {
            $parent = $node->right();

            while ($parent !== null) {
                if (is_null($parent->left())) {
                    return $parent;
                } else {
                    $parent = $parent->left();
                }
            }
        } else {
            $parent = $this->root;
            $successor = null;

            while ($parent !== null) {
                if ($node->value < $parent->value) {
                    $successor = $parent;

                    $parent = $parent->left();
                } else if ($node->value > $parent->value) {
                    $parent = $parent->right();
                } else {
                    break;
                }
            }

            return $successor;
        }
    }

    /**
     * Return the minimum value node or null if tree is empty. O(log V)
     *
     * @return \Rubix\Engine\BinaryNode
     */
    public function min() : ?BinaryNode
    {
        if ($this->isEmpty()) {
            return null;
        }

        $parent = $this->root;

        while ($parent !== null) {
            if (is_null($parent->left())) {
                return $parent;
            } else {
                $parent = $parent->left();
            }
        }

        return $parent;
    }

    /**
     * Return the maximum value node or null if tree is empty. O(log V)
     *
     * @return \Rubix\Engine\BinaryNode
     */
    public function max() : ?BinaryNode
    {
        if ($this->isEmpty()) {
            return null;
        }

        $parent = $this->root;

        while ($parent !== null) {
            if (is_null($parent->right())) {
                return $parent;
            } else {
                $parent = $parent->right();
            }
        }

        return $parent;
    }


    /**
     * Delete a node from the BST and rebalance. O(log V)
     *
     * @param  \Rubix\Engine\BinaryNode  $node
     * @return self
     */
    public function delete(BinaryNode $node) : self
    {
        $parent = $node->parent();

        if (!isset($parent)) {
            if ($node->isLeaf()) {
                $this->root = null;
            } else if (!is_null($node->left())) {
                $this->root = $node->left();
            } else {
                $this->root = $node->right();
            }
        } else {
            if ($node->isLeaf()) {
                if ($node->value > $parent->value) {
                    $parent->detachRight();
                } else {
                    $parent->detachLeft();
                }
            } else if (is_null($node->right()) && !is_null($node->left())) {
                if ($node->value > $parent->value) {
                    $parent->attachRight($node->left());
                    $node->left()->setParent($parent);
                } else {
                    $parent->attachLeft($node->left());
                    $node->left()->setParent($parent);
                }
            } else if (is_null($node->left()) && !is_null($node->right())) {
                if ($node->value > $parent->value) {
                    $parent->attachRight($node->right());
                    $node->right()->setParent($parent);
                } else {
                    $parent->attachLeft($node->right());
                    $node->right()->setParent($parent);
                }
            } else {
                $successor = $this->successor($node);

                $this->delete($successor);

                $node->update($successor->properties());
            }

            while (isset($parent)) {
                $balance = $parent->balance();

                if ($balance > 1 && $parent->left()->balance() >= 0) {
                    $this->rotateRight($parent);
                } else if ($balance < -1 && $parent->right()->balance() <= 0) {
                    $this->rotateLeft($parent);
                } else if ($balance > 1 && $parent->left()->balance() < 0) {
                    $this->rotateLeft($parent->left());
                    $this->rotateRight($parent);
                } else if ($balance < -1 && $parent->right()->balance() > 0) {
                    $this->rotateRight($parent->right());
                    $this->rotateLeft($parent);
                }

                $parent = $parent->parent();
            }
        }

        $this->size--;

        return $this;
    }

    /**
     * Rotates node x to the left as demonstrated in the picture below. O(1)
     *
     *      x                              y
     *     / \        rotate left        /  \
     *   T1   y      – – – - - – >     x    T3
     *       / \                      / \
     *     T2  T3                   T1  T2
     *
     * @param  \Rubix\Engine\BinaryNode  $x
     * @return void
     */
    protected function rotateLeft(BinaryNode $x) : void
    {
        $y = $x->right();

        $y->setParent($x->parent());

        if (is_null($y->parent())) {
            $this->root = $y;
        } else {
            if ($y->parent()->left() === $x) {
                $y->parent()->attachLeft($y);
            } else if ($y->parent()->right() === $x) {
                $y->parent()->attachRight($y);
            }
        }

        $x->attachRight($y->left());

        $y->attachLeft($x);
    }

    /**
     * Rotates node x to the right as demonstrated in the picture below. O(1)
     *
     *      y                             x
     *     / \       rotate right       /  \
     *    x   T3     – – – - - – >    T1    y
     *   / \                               / \
     *  T1  T2                           T2  T3
     *
     * @param  \Rubix\Engine\BinaryNode  $x
     * @return void
     */
    protected function rotateRight(BinaryNode $x) : void
    {
        $y = $x->left();

        $y->setParent($x->parent());

        if (is_null($y->parent())) {
            $this->root = $y;
        } else {
            if ($y->parent()->left() === $x) {
                $y->parent()->attachLeft($y);
            } else if ($y->parent()->right() === $x) {
                $y->parent()->attachRight($y);
            }
        }

        $x->attachLeft($y->right());

        $y->attachRight($x);
    }

    /**
     * Count the number of nodes in the tree.
     *
     * @return int
     */
    public function count() : int
    {
        return $this->size;
    }

    /**
     * Is the tree empty?
     *
     * @return bool
     */
    public function isEmpty() : bool
    {
        return is_null($this->root);
    }
}
