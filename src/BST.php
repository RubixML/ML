<?php

namespace Rubix\Engine;

use InvalidArgumentException;
use Countable;

class BST implements Countable
{
    /**
     * The root node of the binary search tree.
     *
     * @var \Rubix\Engine\BinaryNode|null  $root
     */
    protected $root;

    /**
     * The number of values in the tree.
     *
     * @var int
     */
    protected $size;

    /**
     * @param  array  $values
     * @return void
     */
    public function __construct(array $values = [])
    {
        $this->merge($values);
    }

    /**
     * @return \Rubix\Engine\BinaryNode|null
     */
    public function root() : ?BinaryNode
    {
        return $this->root;
    }

    /**
     * @return int
     */
    public function size() : int
    {
        return $this->size;
    }

    /**
     * Does the given value exist in the BST. O(logV)
     *
     * @param  mixed  $value
     * @return bool
     */
    public function has($value) : bool
    {
        return !is_null($this->find($value));
    }

    /**
     * Insert a node into the BST and balance. O(logV)
     *
     * @param  mixed  $value
     * @param  array  $properties
     * @return \Rubix\Engine\BinaryNode
     */
    public function insert($value, array $properties = []) : BinaryNode
    {
        $node = new BinaryNode($value, $properties);

        if ($this->isEmpty()) {
            $this->root = $node;
        } else {
            $this->_insert($node, $this->root);
        }

        $this->size++;

        return $node;
    }

    /**
     * Recursive function to traverse tree and insert new node.
     *
     * @param  \Rubix\Engine\BinaryNode  $node
     * @param  \Rubix\Engine\BinaryNode  $root
     * @return \Rubix\Engine\BinaryNode
     */
    protected function _insert(BinaryNode $node, BinaryNode $root = null) : BinaryNode
    {
        if (!isset($root)) {
            return $node;
        }

        if ($node->value() < $root->value()) {
            $root->attachLeft($this->_insert($node, $root->left()));
        } else if ($node->value() > $root->value()) {
            $root->attachRight($this->_insert($node, $root->right()));
        } else {
            return $root;
        }

        return $root;
    }

    /**
     * Merge an array of values into the binary search tree. O(NlogV)
     *
     * @param  array  $values
     * @return self
     */
    public function merge(array $values) : BST
    {
        foreach ($values as $value) {
            $this->insert($value);
        }

        return $this;
    }

    /**
     * Find the node matching the given value and return it. O(logV)
     *
     * @param  mixed  $value
     * @return \Rubix\Engine\BinaryNode|null
     */
    public function find($value) : ?BinaryNode
    {
        if (!is_numeric($value) && !is_string($value)) {
            throw new InvalidArgumentException('Search value must be numeric or string type, ' . gettype($value) . ' found.');
        }

        return $this->_find($value, $this->root);
    }

    /**
     * @param  mixed  $value
     * @param  \Rubix\Engine\BinaryNode|null  $root
     * @return \Rubix\Engine\BinaryNode
     */
    protected function _find($value, BinaryNode $root = null) : ?BinaryNode
    {
        if (!isset($root) || $root->value() === $value) {
            return $root;
        }

        if ($root->value() < $value) {
            return $this->_find($value, $root->right());
        } else {
            return $this->_find($value, $root->left());
        }
    }

    /**
     * Find a range of nodes given two values. O(logV)
     *
     * @param  mixed  $start
     * @param  mixed  $end
     * @return array
     */
    public function findRange($start, $end) : array
    {
        if ($start > $end) {
            throw new InvalidArgumentException('Start range value must be less than or equal to end value.');
        }

        $range = [];

        $this->_findRange($start, $end, $this->root, $range);

        return $range;
    }

    /**
     * Recursive function to find all values given a particular range.
     *
     * @param  mixed  $start
     * @param  mixed  $end
     * @param  \Rubix\Engine\BinaryNode|null  $current
     * @param  array  $range
     * @return void
     */
    protected function _findRange($start, $end, BinaryNode $current = null, array &$range = []) : void
    {
        if (!isset($current)) {
            return;
        }

        if ($current->value() > $start) {
            $this->_findRange($start, $end, $current->left(), $range);
        }

        if ($current->value() >= $start && $current->value() <= $end) {
            $range[] = $current->value();
        }

        if ($current->value() < $end) {
            $this->_findRange($start, $end, $current->right(), $range);
        }
    }

    /**
     * Return an array of all the values in the tree.
     *
     * @return array
     */
    public function all()
    {
        $values = [];

        $this->_sort($this->root, $values);

        return $values;
    }

    /**
     * Return all of the values in a sorted array. O(V)
     *
     * @param  \Rubix\Engine\BinaryNode|null  $root
     * @param  array  $values
     * @return void
     */
    protected function _sort(BinaryNode $root = null, &$values = [])
    {
        if (!isset($root)) {
            return $values;
        }

        $this->_sort($root->left(), $values);

        $values[] = $root->value();

        $this->_sort($root->right(), $values);
    }

    /**
     * Return the in order successor of a given node or null if given node is max.
     * O(logV)
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
                if ($node->value() < $parent->value()) {
                    $successor = $parent;

                    $parent = $parent->left();
                } else if ($node->value() > $parent->value()) {
                    $parent = $parent->right();
                } else {
                    break;
                }
            }

            return $successor;
        }
    }

    /**
     * Return the minimum value or null if tree is empty. O(logV)
     *
     * @return mixed
     */
    public function min()
    {
        if ($this->isEmpty()) {
            return null;
        }

        $parent = $this->root;

        while ($parent !== null) {
            if (is_null($parent->left())) {
                return $parent->value();
            } else {
                $parent = $parent->left();
            }
        }

        return null;
    }

    /**
     * Return the maximum value or null if tree is empty. O(logV)
     *
     * @return mixed
     */
    public function max()
    {
        if ($this->isEmpty()) {
            return null;
        }

        $parent = $this->root;

        while ($parent !== null) {
            if (is_null($parent->right())) {
                return $parent->value();
            } else {
                $parent = $parent->right();
            }
        }

        return null;
    }


    /**
     * Delete a value from the BST.
     *
     * @param  mixed  $value
     * @return self
     */
    public function delete($value) : BST
    {
        $this->_delete($value, $this->root);

        $this->size--;

        return $this;
    }

    public function _delete($value, BinaryNode $root = null)
    {
        if (!isset($root)) {
            return $root;
        }

        if ($value < $root->value()) {
            $root->attachLeft($this->_delete($value, $root->left()));
        } else if ($value > $root->value()) {
            $root->attachRight($this->_delete($value, $root->right()));
        } else {
            if (is_null($root->left())) {
                return $root->right();
            } else if (is_null($root->right())) {
                return $root->left();
            }
        }

        return $root;
    }

    /**
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
        return !isset($this->root);
    }
}
