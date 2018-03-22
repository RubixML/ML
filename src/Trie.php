<?php

namespace Rubix\Engine;

use Countable;
use SplStack;

class Trie extends Tree implements Countable
{
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
        $this->size = 0;

        parent::__construct(new GraphNode('*'));

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
     * Is a word present in the trie? O(L)
     *
     * @param  string  $word
     * @return bool
     */
    public function has(string $word) : bool
    {
        $path = $this->find($word);

        if (isset($path)) {
            return $path->last()->get('word', false);
        }

        return false;
    }

    /**
     * Insert a word into the trie and return a path of nodes. O(L)
     *
     * @param  string  $word
     * @return \Rubix\Engine\Path
     */
    public function insert(string $word) : Path
    {
        $path = new Path();

        $current = $this->root;

        foreach (str_split(strtolower(trim($word))) as $letter) {
            if ($current->edges()->has($letter)) {
                $current = $current->edges()->get($letter)->node();
            } else {
                $current = $current->attach(new GraphNode($letter))->node();
            }

            $path->append($current);
        }

        if (!$current->word) {
            $current->set('word', true);

            $this->size++;
        }

        return $path;
    }

    /**
     * Merge an array of words into the trie at once. O(N*L)
     *
     * @param  array  $words
     * @return self
     */
    public function merge(array $words) : self
    {
        foreach ($words as $word) {
            $this->insert($word);
        }

        return $this;
    }

    /**
     * Find a path of given prefix or null if one does not exist. O(L)
     *
     * @param  string  $prefix
     * @return \Rubix\Engine\Path
     */
    public function find(string $prefix) : ?Path
    {
        $path = new Path();

        $current = $this->root;

        foreach (str_split(strtolower(trim($prefix))) as $letter) {
            if ($current->edges()->has($letter)) {
                $current = $current->edges()->get($letter)->node();

                $path->append($current);
            } else {
                return null;
            }
        }

        return $path;
    }

    /**
     * Remove a word from the trie and trim the suffix. O(L)
     *
     * @param  string  $word
     * @return self
     */
    public function delete(string $word) : self
    {
        $stack = new SplStack();

        $current = $this->root;

        foreach (str_split(strtolower(trim($word))) as $letter) {
            if ($current->edges()->has($letter)) {
                $current = $current->edges()->get($letter)->node();

                $stack->push($current);
            } else {
                return $this;
            }
        }

        if ($current->word) {
            $current->set('word', false);

            $this->size--;
        }

        while (!$stack->isEmpty()) {
            $current = $stack->pop();

            if (!$current->word && $current->isLeaf()) {
                if ($stack->valid()) {
                    $stack->top()->edges()->remove($current->id());
                }
            } else {
                break;
            }
        }

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
