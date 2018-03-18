<?php

namespace Rubix\Engine;

use InvalidArgumentException;
use Countable;
use SplStack;

class BST extends Tree implements Countable
{
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
        parent::__construct();

        $this->size = 0;
    }

    /**
     * @return int
     */
    public function size() : int
    {
        return $this->size;
    }

    /**
     * The height of the tree.
     *
     * @return int
     */
    public function height() : int
    {
        return $this->root->height();
    }

    /**
     * The balance factor of the tree.
     *
     * @return int
     */
    public function balance() : int
    {
        return $this->root->balance();
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
     * Insert a node into the BST. ϴ(logV), O(V)
     *
     * @param  mixed  $value
     * @param  array  $properties
     * @return \Rubix\Engine\BinaryNode
     */
    public function insert($value, array $properties = []) : BinaryNode
    {
        $node = new BinaryNode($value, $properties);

        $this->_insert($node);

        return $node;
    }

    /**
     * Helper function to insert a node into the tree.
     *
     * @param  \Rubix\Engine\BinaryNode  $node
     * @return void
     */
    protected function _insert(BinaryNode $node) : void
    {
        if ($this->isEmpty()) {
            $this->root = $node;
        } else {
            $current = $this->root;

            while (isset($current)) {
                if ($current->value() > $node->value()) {
                    if (is_null($current->left())) {
                        $current->attachLeft($node);
                        break;
                    } else {
                        $current = $current->left();
                    }
                } else {
                    if (is_null($current->right())) {
                        $current->attachRight($node);
                        break;
                    } else {
                        $current = $current->right();
                    }
                }
            }
        }

        $this->size++;
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
     * @throws \InvalidArgumentException
     * @return \Rubix\Engine\BinaryNode|null
     */
    public function find($value) : ?BinaryNode
    {
        if (!is_numeric($value) && !is_string($value)) {
            throw new InvalidArgumentException('Value must be a string or numeric type, ' . gettype($value) . ' found.');
        }

        $current = $this->root;

        while (isset($current)) {
            if ($current->value() === $value) {
                break;
            } else if ($current->value() > $value) {
                $current = $current->left();
            } else {
                $current = $current->right();
            }
        }

        return $current;
    }

    /**
     * Find a range of nodes with values between start and end values. Returns
     * null if range not found.
     *
     * @param  mixed  $start
     * @param  mixed  $end
     * @throws \InvalidArgumentException
     * @return \Rubix\Engine\Path|null
     */
    public function findRange($start, $end) : ?Path
    {
        if ($start > $end) {
            throw new InvalidArgumentException('Start value must be less than or equal to end value.');
        }

        $path = new Path();

        $this->_findRange($start, $end, $this->root, $path);

        return !$path->isEmpty() ? $path : null;
    }

    /**
     * Recursive function to find all nodes with values given in a certain range.
     *
     * @param  mixed  $start
     * @param  mixed  $end
     * @param  \Rubix\Engine\BinaryNode  $current
     * @param  \Rubix\Engine\Path  $path
     * @return void
     */
    protected function _findRange($start, $end, BinaryNode $current = null, Path $path) : void
    {
        if (!isset($current)) {
            return;
        }

        if ($current->value() > $start) {
            $this->_findRange($start, $end, $current->left(), $path);
        }

        if ($current->value() >= $start && $current->value <= $end) {
            $path->append($current);
        }

        if ($current->value() < $end) {
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
        if ($this->isEmpty()) {
            return null;
        }

        $stack = new SplStack();
        $path = new Path();

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
     * Return the in order predecessor of a given node or null if given node is min.
     *
     * @param  mixed  $value
     * @return \Rubix\Engine\BinaryNode|null
     */
    public function predecessor($value) : ?BinaryNode
    {
        $current = $this->find($value);

        if (!isset($current)) {
            return null;
        }

        return $this->_predecessor($current);
    }

    /**
     * @param  \Rubix\Engine\BinaryNode  $node
     * @return \Rubix\Engine\BinaryNode|null
     */
    protected function _predecessor(BinaryNode $node) : ?BinaryNode
    {
        if (is_null($node->left())) {
            $parent = $node->parent();

            while (isset($parent) && $node === $parent->left()) {
                $node = $parent;

                $parent = $parent->parent();
            }

            return $parent;
        }

        return $this->_max($node->left());
    }

    /**
     * Return the in order successor of a given node or null if given node is max.
     *
     * @param  \Rubix\Engine\BinaryNode  $node
     * @return \Rubix\Engine\BinaryNode|null
     */
    public function successor($value) : ?BinaryNode
    {
        $current = $this->find($value);

        if (!isset($current)) {
            return null;
        }

        return $this->_successor($current);
    }

    /**
     * @param  \Rubix\Engine\BinaryNode  $node
     * @return \Rubix\Engine\BinaryNode|null
     */
    protected function _successor(BinaryNode $node = null) : ?BinaryNode
    {
        if (is_null($node->right())) {
            $parent = $node->parent();

            while (isset($parent) && $node === $parent->right()) {
                $node = $parent;

                $parent = $parent->parent();
            }

            return $parent;
        }

        return $this->_min($node->right());
    }

    /**
     * Return the minimum value node or null if tree is empty. O(log V)
     *
     * @return \Rubix\Engine\BinaryNode
     */
    public function min() : ?BinaryNode
    {
        return isset($this->root) ? $this->_min($this->root) : null;
    }

    /**
     * @param  \Rubix\Engine\BinaryNode  $root
     * @return \Rubix\Engine\BinaryNode
     */
    protected function _min(BinaryNode $root) : BinaryNode
    {
        while (isset($root)) {
            if (is_null($root->left())) {
                return $root;
            } else {
                $root = $root->left();
            }
        }

        return $root;
    }

    /**
     * Return the maximum value node or null if tree is empty. O(log V)
     *
     * @return \Rubix\Engine\BinaryNode
     */
    public function max() : ?BinaryNode
    {
        return isset($this->root) ? $this->_max($this->root) : null;
    }

    /**
     * @param  \Rubix\Engine\BinaryNode  $root
     * @return \Rubix\Engine\BinaryNode
     */
    public function _max(BinaryNode $root) : BinaryNode
    {
        while (isset($root)) {
            if (is_null($root->right())) {
                return $root;
            } else {
                $root = $root->right();
            }
        }

        return $root;
    }

    /**
     * Delete a single node from the tree by value.
     *
     * @param  mixed  $value
     * @return self
     */
    public function delete($value) : self
    {
        $node = $this->find($value);

        if (!isset($node)) {
            return $this;
        }

        $this->_delete($node);

        return $this;
    }

    /**
     * Helper function to delete a node from the BST and rebalance. O(log V)
     *
     * @param  \Rubix\Engine\BinaryNode  $node
     * @return void
     */
    protected function _delete(BinaryNode $node) : void
    {
        $parent = $node->parent();

        if (!isset($parent)) {
            $successor = $this->_successor($node);

            $this->root->changeValue($successor->value())
                ->update($successor->properties());

            $this->_delete($successor);
        } else {
            if ($node->isLeaf()) {
                if ($parent->left() === $node) {
                    $parent->detachLeft();
                } else if ($parent->right() === $node) {
                    $parent->detachRight();
                }
            } else if (is_null($node->right()) && !is_null($node->left())) {
                if ($parent->left() === $node) {
                    $parent->attachLeft($node->left());
                } else if ($parent->right() === $node) {
                    $parent->attachRight($node->left());
                }
            } else if (is_null($node->left()) && !is_null($node->right())) {
                if ($parent->left() === $node) {
                    $parent->attachLeft($node->right());
                } else if ($parent->right() === $node) {
                    $parent->attachRight($node->right());
                }
            } else {
                $successor = $this->_successor($node);

                $this->_delete($successor);

                $node->changeValue($successor->value())
                    ->update($successor->properties());
            }
        }

        $this->size--;
    }

    /**
     * Count the number of nodes in the tree. Alias of size().
     *
     * @return int
     */
    public function count() : int
    {
        return $this->size();
    }
}
