<?php

namespace Rubix\Engine;

use Countable;

class Trie implements Countable
{
    /**
     * The root node of the trie.
     *
     * @var \Rubix\Engine\Node|null  $root
     */
    protected $root;

    /**
     * The size of the trie in words.
     *
     * @var int
     */
    protected $size;

    /**
     * @param  array  $words
     * @return void
     */
    public function __construct(array $words = [])
    {
        $this->root = new Node('*', ['parent' => null]);
        $this->size = 0;

        $this->merge($words);
    }

    /**
     * @return \Rubix\Engine\Node|null
     */
    public function root() : ?Node
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
     * Is a word present in the trie? O(L)
     *
     * @param  string  $word
     * @return bool
     */
    public function has(string $word) : bool
    {
        $current = $this->find($word);

        if (isset($current)) {
            return $current->word ? true : false;
        }

        return false;
    }

    /**
     * Insert a word into the trie. O(L)
     *
     * @param  string  $word
     * @return \Rubix\Engine\Node
     */
    public function insert(string $word) : Node
    {
        $current = $this->root;

        foreach (str_split(strtolower(trim($word))) as $key) {
            if ($current->edges()->has($key)) {
                $current = $current->edges()->get($key)->node();
            } else {
                $current = $current->attach(new Node($key, ['parent' => $current]))->node();
            }
        }

        if ($current->get('word', false) === false) {
            $current->set('word', true);

            $this->size++;
        }

        return $current;
    }

    /**
     * Merge an array of words into the trie at once. O(N*L)
     *
     * @param  array  $words
     * @return self
     */
    public function merge(array $words) : Trie
    {
        foreach ($words as $word) {
            $this->insert($word);
        }

        return $this;
    }

    /**
     * Find a prefix node by key. O(L)
     *
     * @param  string  $prefix
     * @return \Rubix\Engine\Node|null
     */
    public function find(string $prefix) : ?Node
    {
        $current = $this->root;

        foreach (str_split(strtolower(trim($prefix))) as $key) {
            if ($current->edges()->has($key)) {
                $current = $current->edges()->get($key)->node();
            } else {
                return null;
            }
        }

        return $current;
    }

    /**
     * Delete a word from the trie. O(L)
     *
     * @param  string  $word
     * @return self
     */
    public function delete(string $word) : Trie
    {
        $current = $this->find($word);

        if (is_null($current)) {
            return $this;
        }

        $current->set('word', false);

        while ($current !== null) {
            if ($current->get('word', false) === false && $current->isLeaf()) {
                $current->parent->edges()->remove($current->id());

                $current = $current->parent;
            } else {
                break;
            }
        }

        $this->size--;

        return $this;
    }

    /**
     * @return int
     */
    public function count() : int
    {
        return $this->size;
    }

    /**
     * Is the trie empty?
     *
     * @return bool
     */
    public function isEmpty() : bool
    {
        return $this->size <= 0;
    }
}
