<?php

namespace Rubix\Engine;

use Countable;

class Trie extends Tree implements Countable
{
    /**
     * The size of the trie in words.
     *
     * @var int
     */
    public $size;

    /**
     * @param  array  $words
     * @return void
     */
    public function __construct(array $words = [])
    {
        $this->size = 0;

        parent::__construct(new Node('*', ['parent' => null]));

        $this->merge($words);
    }

    /**
     * @return int
     */
    public function size() : int
    {
        return $this->size;
    }

    /**
     * Is a word present in the trie?
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
     * Is the given prefix present in the trie?
     *
     * @param  string  $word
     * @return bool
     */
    public function hasPrefix(string $prefix) : bool
    {
        return !is_null($this->find($prefix));
    }

    /**
     * Insert a word into the trie.
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

        if ($current->word !== true) {
            $current->set('word', true);

            $this->size++;
        }

        return $current;
    }

    /**
     * Merge an array of words into the trie at once.
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
     * Find the given prefix in the trie.
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
     * Delete a word from the trie.
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
            if ($current->word !== true && $current->isLeaf()) {
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
}
