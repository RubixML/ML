<?php

namespace Rubix\Engine;

class GADDAG
{
    /**
     * The root node of the GADDDAG.
     *
     * @var \Rubix\Engine\Node  $root
     */
    protected $root;

    /**
     * @param  array  $words
     * @return void
     */
    public function __construct(array $words = [])
    {
        $this->root = new Node('*');

        $this->merge($words);
    }

    /**
     * @return \Rubix\Engine\Node
     */
    public function root() : Node
    {
        return $this->root;
    }

    /**
     * Is a word present in the GADDAG? O(L)
     *
     * @param  string  $word
     * @return bool
     */
     public function has(string $word) : bool
     {
         $current = $this->root;

         foreach (str_split(strtolower(strrev(trim($word)) . '|$')) as $key) {
             if ($current->edges()->has($key)) {
                 $current = $current->edges()->get($key)->node();
             } else {
                 return false;
             }
         }

         return true;
     }

    /**
     * Insert a word into the GADDAG. Creates a path from the root node for each
     * letter of the word making it accessable from anywhere in the string. O(L^2)
     *
     * @param  string  $word
     * @return self
     */
    public function insert(string $word) : GADDAG
    {
        $word = strtolower(trim($word));
        $length = strlen($word);
        $path = [];

        $current = $this->root;

        foreach (range(1, $length) as $i) {
            $prefix = substr($word, 0, $i);
            $suffix = $i !== $length ? substr($word, $i, $length - $i) : '';

            $keys = str_split(strrev($prefix) . '|' . $suffix . '$');

            $current = $this->root;
            $truncate = false;

            foreach ($keys as $j => $key) {
                if ($truncate === true && count($path) > $j) {
                    $current = $current->attach($path[$j])->node();

                    break;
                }

                if ($current->edges()->has($key)) {
                    $current = $current->edges()->get($key)->node();
                } else {
                    $current = $current->attach(new Node($key))->node();
                }

                if (count($path) === $j) {
                    $path[$j] = $current;
                }

                if ($key === '|') {
                    $truncate = true;
                }
            }
        }

        return $this;
    }

    /**
     * Merge an array of words into the GADDAG.
     *
     * @param  array  $words
     * @return self
     */
    public function merge(array $words) : GADDAG
    {
        foreach ($words as $word) {
            $this->insert($word);
        }

        return $this;
    }
}
