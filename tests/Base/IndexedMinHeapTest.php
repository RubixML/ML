<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Base;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\IndexedMinHeap;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\TestCase;

use function count;
use function mt_srand;
use function is_nan;
use function rand;

#[Group('Base')]
#[CoversClass(IndexedMinHeap::class)]
class IndexedMinHeapTest extends TestCase
{
    protected IndexedMinHeap $heap;

    protected function setUp() : void
    {
        $this->heap = new IndexedMinHeap();
    }

    #[Test]
    public function emptyHeap() : void
    {
        $this->assertSame(0, $this->heap->count());
        $this->assertNull($this->heap->extractMin());
        $this->assertFalse($this->heap->has(0));
        $this->assertNull($this->heap->key(0));
    }

    #[Test]
    public function buildExtractionOrder() : void
    {
        $this->heap->build([2, 9, 0, 5, 3], [
            2 => 4.0,
            9 => 1.5,
            0 => 2.5,
            5 => 10.0,
            3 => 0.25,
        ]);

        $this->assertSame(5, $this->heap->count());

        $this->assertSame([0.25, 3], $this->heap->extractMin());
        $this->assertSame([1.5, 9], $this->heap->extractMin());
        $this->assertSame([2.5, 0], $this->heap->extractMin());
        $this->assertSame([4.0, 2], $this->heap->extractMin());
        $this->assertSame([10.0, 5], $this->heap->extractMin());
        $this->assertNull($this->heap->extractMin());
        $this->assertSame(0, $this->heap->count());
    }

    #[Test]
    public function buildIsIndependentOfInsertionOrder() : void
    {
        $keys = [3 => 7.5, 41 => 2.0, 17 => 2.0, 8 => 99.0];

        $a = new IndexedMinHeap();
        $a->build([3, 41, 17, 8], $keys);

        $b = new IndexedMinHeap();
        $b->build([8, 17, 41, 3], $keys);

        $this->assertEquals($a->extractMin(), $b->extractMin());
        $this->assertEquals($a->extractMin(), $b->extractMin());
        $this->assertEquals($a->extractMin(), $b->extractMin());
        $this->assertEquals($a->extractMin(), $b->extractMin());
    }

    #[Test]
    public function buildRejectsMissingKey() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->heap->build([1, 2], [1 => 1.0]);
    }

    #[Test]
    public function buildEmpty() : void
    {
        $this->heap->build([], []);

        $this->assertSame(0, $this->heap->count());
        $this->assertNull($this->heap->extractMin());
    }

    #[Test]
    public function setInsertsThenAdjustsUpDown() : void
    {
        $this->heap->build([1, 2], [
            1 => 10.0,
            2 => 20.0,
        ]);

        $this->heap->set(3, 5.0);

        $this->assertSame([5.0, 3], $this->heap->extractMin());

        $this->heap->set(1, 100.0);

        $this->assertSame([20.0, 2], $this->heap->extractMin());
        $this->assertSame([100.0, 1], $this->heap->extractMin());

        $this->heap->set(7, 0.5);

        $this->assertSame([0.5, 7], $this->heap->extractMin());
    }

    #[Test]
    public function removePreservesOrder() : void
    {
        $this->heap->build([1, 2, 3, 4, 5, 6], [
            1 => 6.0,
            2 => 5.0,
            3 => 4.0,
            4 => 3.0,
            5 => 2.0,
            6 => 1.0,
        ]);

        $this->heap->remove(6);

        $this->assertSame(5, $this->heap->count());
        $this->assertFalse($this->heap->has(6));

        $this->assertSame([2.0, 5], $this->heap->extractMin());
        $this->assertSame([3.0, 4], $this->heap->extractMin());
        $this->assertSame([4.0, 3], $this->heap->extractMin());
        $this->assertSame([5.0, 2], $this->heap->extractMin());
        $this->assertSame([6.0, 1], $this->heap->extractMin());

        $this->heap->remove(1);

        $this->assertSame(0, $this->heap->count());

        $this->heap->remove(99);
    }

    #[Test]
    public function tiesBreakByPairCode() : void
    {
        $this->heap->build([42, 7, 10], [
            42 => 3.0,
            7 => 3.0,
            10 => 3.0,
        ]);

        $this->assertSame([3.0, 7], $this->heap->extractMin());
        $this->assertSame([3.0, 10], $this->heap->extractMin());
        $this->assertSame([3.0, 42], $this->heap->extractMin());
    }

    #[Test]
    public function nanKeysRankLastAndBreakByPairCode() : void
    {
        $this->heap->build([8, 5, 2], [
            8 => NAN,
            5 => NAN,
            2 => 1.0,
        ]);

        $this->assertSame([1.0, 2], $this->heap->extractMin());

        $result = $this->heap->extractMin();

        $this->assertTrue(is_nan($result[0]));
        $this->assertSame(5, $result[1]);

        $result = $this->heap->extractMin();

        $this->assertTrue(is_nan($result[0]));
        $this->assertSame(8, $result[1]);
    }

    #[Test]
    public function allNanKeysBreakByPairCode() : void
    {
        $this->heap->build([8, 5, 2], [
            8 => NAN,
            5 => NAN,
            2 => NAN,
        ]);

        foreach ([2, 5, 8] as $pair) {
            $result = $this->heap->extractMin();

            $this->assertTrue(is_nan($result[0]));
            $this->assertSame($pair, $result[1]);
        }
    }

    #[Test]
    public function randomizedAgainstBruteForce() : void
    {
        mt_srand(0);

        $heap = new IndexedMinHeap();
        $state = [];

        $maxPair = 32;

        for ($iter = 0; $iter < 500; ++$iter) {
            $op = rand(0, 2);

            if ($op === 0) {
                $pair = rand(1, $maxPair);
                $key = rand(0, 100) * 0.25;

                $state[$pair] = $key;

                $heap->set($pair, $key);
            } elseif ($op === 1) {
                $pair = rand(1, $maxPair);

                $heap->remove($pair);

                unset($state[$pair]);
            } else {
                if (count($state) < 1) {
                    $pair = rand(1, $maxPair);
                    $key = rand(0, 100) * 0.25;

                    $state[$pair] = $key;

                    $heap->set($pair, $key);

                    $this->assertSame(1, $heap->count());

                    $expected = $pair;
                } else {
                    $expected = null;
                    $minKey = null;

                    foreach ($state as $pair => $key) {
                        if ($minKey === null or $key < $minKey or ($key == $minKey and $pair < $expected)) {
                            $minKey = $key;
                            $expected = $pair;
                        }
                    }
                }

                $this->assertSame(
                    [$heap->key($expected), $expected],
                    $heap->extractMin()
                );

                unset($state[$expected]);
            }

            $this->assertSame(count($state), $heap->count());
        }
    }
}
