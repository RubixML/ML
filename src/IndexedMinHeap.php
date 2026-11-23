<?php

declare(strict_types=1);

namespace Rubix\ML;

use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;
use function array_key_exists;
use function array_pop;
use function is_nan;

/**
 * Indexed Min Heap
 *
 * A binary min-heap of integer pair codes, ordered by a float key assigned to
 * each pair. Unlike a plain priority queue, entries are addressed by their
 * pair code so an existing key can be adjusted in place or the entry removed
 * outright, without the stale copies that successive updates of a lazy
 * priority queue would accumulate. The total order — smaller key first, pair
 * code breaking all remaining ties, including `NAN` — is fully deterministic.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @internal
 */
class IndexedMinHeap
{
    /**
     * The pair codes heap-ordered by the key currently assigned to them.
     *
     * @var list<int>
     */
    protected array $heap = [];

    /**
     * The position of each pair code in the heap.
     *
     * @var array<int, int>
     */
    protected array $pos = [];

    /**
     * The key currently assigned to each pair code.
     *
     * @var array<int, float>
     */
    protected array $keys = [];

    /**
     * @var int
     */
    protected int $size = 0;

    /**
     * Seed the heap with a batch of pair codes and their keys, heapifying in
     * O(n) rather than inserting one at a time.
     *
     * @param list<int> $pairs
     * @param array<int, float> $keys
     * @throws InvalidArgumentException
     */
    public function build(array $pairs, array $keys) : void
    {
        $this->size = count($pairs);

        if ($this->size < 1) {
            $this->heap = [];
            $this->pos = [];
            $this->keys = [];

            return;
        }

        $this->heap = $pairs;
        $this->keys = $keys;
        $this->pos = [];

        foreach ($pairs as $index => $pair) {
            if (!array_key_exists($pair, $keys)) {
                throw new InvalidArgumentException('No key was given for pair'
                    . " code $pair.");
            }

            $this->pos[$pair] = $index;
        }

        for ($i = ($this->size >> 1) - 1; $i >= 0; --$i) {
            $this->siftDown($i);
        }
    }

    /**
     * Assign a key to a pair code, inserting it if new or adjusting it —
     * sifting up or down as required — if already present.
     *
     * @internal
     *
     * @param int $pair
     * @param float $key
     */
    public function set(int $pair, float $key) : void
    {
        $this->keys[$pair] = $key;

        if (array_key_exists($pair, $this->pos)) {
            $this->reseat($this->pos[$pair]);

            return;
        }

        $this->pos[$pair] = $this->size;
        $this->heap[] = $pair;

        ++$this->size;

        $this->siftUp($this->size - 1);
    }

    /**
     * Remove the entry for a pair code, if present.
     *
     * @internal
     *
     * @param int $pair
     */
    public function remove(int $pair) : void
    {
        if (!array_key_exists($pair, $this->pos)) {
            return;
        }

        $i = $this->pos[$pair];
        $last = $this->size - 1;
        $moved = $this->heap[$last];

        unset($this->pos[$pair], $this->keys[$pair]);

        array_pop($this->heap);

        --$this->size;

        if ($i === $last) {
            return;
        }

        $this->heap[$i] = $moved;
        $this->pos[$moved] = $i;

        $this->reseat($i);
    }

    /**
     * Extract the pair code with the currently minimum key, or `null` if the
     * heap is empty.
     *
     * @internal
     *
     * @return array{0: float, 1: int}|null
     */
    public function extractMin() : ?array
    {
        if ($this->size < 1) {
            return null;
        }

        $min = $this->heap[0];
        $key = $this->keys[$min];
        $moved = $this->heap[$this->size - 1];

        unset($this->pos[$min], $this->keys[$min]);

        array_pop($this->heap);

        --$this->size;

        if ($this->size < 1) {
            return [$key, $min];
        }

        $this->heap[0] = $moved;
        $this->pos[$moved] = 0;

        $this->siftDown(0);

        return [$key, $min];
    }

    /**
     * Return the number of entries in the heap.
     *
     * @internal
     *
     * @return int
     */
    public function count() : int
    {
        return $this->size;
    }

    /**
     * Is the pair code currently a member of the heap.
     *
     * @internal
     *
     * @param int $pair
     * @return bool
     */
    public function has(int $pair) : bool
    {
        return array_key_exists($pair, $this->pos);
    }

    /**
     * Return the key currently assigned to a pair code, if any.
     *
     * @internal
     *
     * @param int $pair
     * @return float|null
     */
    public function key(int $pair) : ?float
    {
        if (!array_key_exists($pair, $this->keys)) {
            return null;
        }

        return $this->keys[$pair];
    }

    /**
     * Is $a ranked strictly before $b in the heap's total order — smaller
     * key first, pair code breaking all exact ties. A `NAN` key ranks after
     * every finite value, so it only surfaces once no finite key remains.
     *
     * @internal
     *
     * @param int $a
     * @param int $b
     * @return bool
     */
    protected function lessThan(int $a, int $b) : bool
    {
        $ka = $this->keys[$a];
        $kb = $this->keys[$b];

        $nanA = is_nan($ka);
        $nanB = is_nan($kb);

        if ($nanA xor $nanB) {
            return $nanB;
        }

        if ($nanA) {
            return $a < $b;
        }

        if ($ka < $kb) {
            return true;
        }

        if ($kb < $ka) {
            return false;
        }

        return $a < $b;
    }

    /**
     * Restore the heap invariant around index $i after its key changed or its
     * occupant moved, sifting in whichever direction is required.
     *
     * @internal
     *
     * @param int $i
     */
    protected function reseat(int $i) : void
    {
        if ($i < 1) {
            $this->siftDown($i);

            return;
        }

        $p = ($i - 1) >> 1;

        if ($this->lessThan($this->heap[$p], $this->heap[$i])) {
            $this->siftDown($i);

            return;
        }

        $this->siftUp($i);
    }

    /**
     * Restore the heap invariant by bubbling the entry at $i up the tree.
     *
     * @internal
     *
     * @param int $i
     */
    protected function siftUp(int $i) : void
    {
        while ($i > 0) {
            $p = ($i - 1) >> 1;

            if ($this->lessThan($this->heap[$p], $this->heap[$i])) {
                break;
            }

            $this->swap($i, $p);

            $i = $p;
        }
    }

    /**
     * Restore the heap invariant by sinking the entry at $i down the tree.
     *
     * @internal
     *
     * @param int $i
     */
    protected function siftDown(int $i) : void
    {
        while (true) {
            $l = ($i << 1) + 1;

            if ($l >= $this->size) {
                break;
            }

            $r = $l + 1;
            $s = $l;

            if ($r < $this->size and $this->lessThan($this->heap[$r], $this->heap[$l])) {
                $s = $r;
            }

            if ($this->lessThan($this->heap[$i], $this->heap[$s])) {
                break;
            }

            $this->swap($i, $s);

            $i = $s;
        }
    }

    /**
     * Exchange the entries at $i and $j, updating the position map.
     *
     * @internal
     *
     * @param int $i
     * @param int $j
     */
    protected function swap(int $i, int $j) : void
    {
        $a = $this->heap[$i];
        $b = $this->heap[$j];

        $this->heap[$i] = $b;
        $this->heap[$j] = $a;

        $this->pos[$a] = $j;
        $this->pos[$b] = $i;
    }
}
